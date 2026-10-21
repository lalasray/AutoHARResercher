import json

import numpy as np
import pytest
import torch

from autohar_researcher.datasets import make_task
from autohar_researcher.harness import HARHarness
from autohar_researcher.synthetic import MCPGenerator
from autohar_researcher.types import Pipeline


def small_pipeline():
    return Pipeline.from_dict({'data': {'window_seconds': 1, 'overlap': 0},
        'model': {'architecture': 'cnn', 'hidden_dim': 4, 'depth': 1,
                  'max_epochs': 2, 'patience': 1, 'batch_size': 64}})


def test_training_is_source_only_and_final_test_once(manifest, tmp_path, monkeypatch):
    torch.set_num_threads(1)
    task = make_task(manifest, 'loso', 'uci_har', '0')
    paths = {r.path for r in task.test}
    original_load = np.load
    accessed = []
    def spy(path, *args, **kwargs):
        accessed.append(path)
        return original_load(path, *args, **kwargs)
    monkeypatch.setattr(np, 'load', spy)
    harness = HARHarness(task, 13, tmp_path / 'output', device='cpu')
    p = small_pipeline()
    harness.preflight(p)
    result = harness.evaluate(p)
    assert 0 <= result.score <= 1
    assert not set(accessed) & paths
    report = harness.finalize(p)
    assert set(accessed) & paths == paths
    assert report['refit_epochs'] == result.best_epoch
    with pytest.raises(RuntimeError, match='once'):
        harness.finalize(p)
    with pytest.raises(RuntimeError, match='closed'):
        harness.evaluate(p)
    fresh = HARHarness(task, 13, tmp_path / 'output', device='cpu')
    with pytest.raises(RuntimeError, match='once'):
        fresh.finalize(p)


def test_custom_model_code_is_executed(manifest, tmp_path):
    task = make_task(manifest, 'loso', 'uci_har', '0')
    harness = HARHarness(task, 13, tmp_path / 'custom', device='cpu')
    p = small_pipeline()
    p.code['research/components/model.py'] = '''import torch
class Model(torch.nn.Module):
    def __init__(self, channels, classes):
        super().__init__()
        self.layer = torch.nn.Linear(channels, classes)
    def forward(self, x):
        return self.layer(x.mean(dim=1))
def build_model(input_dim, num_classes, params):
    return Model(params['input_channels'], num_classes)
'''
    harness.preflight(p)
    assert harness._build_model(p).__class__.__name__ == 'Model'


def test_incompatible_shape_rejected_before_training(manifest, tmp_path):
    task = make_task(manifest, 'loso', 'uci_har', '0')
    harness = HARHarness(task, 13, tmp_path / 'invalid', device='cpu')
    p = small_pipeline()
    p.code['research/components/data.py'] = 'def preprocess(x):\n    return x[:, :, :1]\n'
    with pytest.raises(ValueError, match='preserve'):
        harness.preflight(p)


def test_synthesis_has_no_proxy_fallback(manifest, tmp_path):
    task = make_task(manifest, 'loso', 'uci_har', '0')
    harness = HARHarness(task, 13, tmp_path / 'synthesis', device='cpu')
    p = small_pipeline()
    p.augmentation['synthetic'] = [{'generator': 'text', 'label': 'walking', 'count': 3, 'prompt': 'walk'}]
    with pytest.raises(ValueError, match='not configured'):
        harness.preflight(p)


def test_mcp_sample_contract(monkeypatch):
    generator = MCPGenerator({'text': {}})
    async def fake_call(spec, args):
        assert args['sampling_hz'] == 50
        return {'samples': np.zeros((2, 50, 6)).tolist()}
    monkeypatch.setattr(generator, '_call', fake_call)
    assert generator.generate({'generator': 'text', 'label': 'walking', 'count': 2, 'prompt': 'walk'}, 50, 6).shape == (2, 50, 6)
    with pytest.raises(ValueError, match='contract'):
        generator.generate({'generator': 'text', 'label': 'walking', 'count': 3, 'prompt': 'walk'}, 50, 6)


def test_lodo_minimum_class_coverage(manifest, tmp_path):
    harness = HARHarness(make_task(manifest, 'lodo', 'uci_har'), 13, tmp_path / 'lodo', device='cpu')
    with pytest.raises(ValueError, match='100 training windows'):
        harness.preflight(small_pipeline())


@pytest.mark.parametrize('architecture', [
    'deepconvlstm', 'deepconvlstmattention', 'bilstm', 'bilstmattention', 'hart', 'mobilehart',
    'spectra', 'tinyhar', 'tinierhar', 'cnn', 'cnn_gru', 'gru', 'tiny_transformer',
])
def test_architectures_use_real_sequence_tensors(architecture):
    import importlib
    torch.set_num_threads(1)
    module = importlib.import_module(f'research.modules.architectures.{architecture}')
    from research.modules.architectures.presets import model_defaults
    steps = 128 if architecture == 'mobilehart' else 50
    model = module.build_model(steps * 6, 5, dict(model_defaults(architecture), input_channels=6))
    model.eval()
    with torch.no_grad():
        result = model(torch.zeros(2, steps, 6))
    assert result.shape == (2, 5)
    assert torch.isfinite(result).all()


def test_synthetic_ratio_is_relative_to_real_class_count(manifest, tmp_path):
    from types import SimpleNamespace
    task = make_task(manifest, 'loso', 'uci_har', '0')
    generator = SimpleNamespace(servers={'text': {}}, generate=lambda request, length, channels, seed:
                                np.zeros((request['count'], length, channels), dtype=np.float32))
    harness = HARHarness(task, 13, tmp_path / 'ratios', generator=generator, device='cpu')
    p = small_pipeline()
    _, real_y, _ = harness._training_arrays(p, fit=False)
    real_class_count = np.sum(real_y == 0)
    p.augmentation['synthetic'] = [
        {'generator': 'text', 'label': 'walking', 'count': 100, 'ratio': 1, 'prompt': 'walk'},
        {'generator': 'text', 'label': 'walking', 'count': 100, 'ratio': 1, 'prompt': 'walk slowly'},
    ]
    _, y, _ = harness._training_arrays(p, fit=False)
    assert np.sum(y == 0) == real_class_count * 3
    assert len(y) == len(real_y) + 2 * real_class_count


def test_requested_cuda_fails_without_gpu(manifest, tmp_path, monkeypatch):
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    with pytest.raises(RuntimeError, match='CUDA training requested'):
        HARHarness(make_task(manifest, 'loso', 'uci_har', '0'), 13, tmp_path, device='cuda')


def test_requested_cuda_index_is_validated(manifest, tmp_path, monkeypatch):
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
    monkeypatch.setattr(torch.cuda, 'device_count', lambda: 1)
    with pytest.raises(ValueError, match='index'):
        HARHarness(make_task(manifest, 'loso', 'uci_har', '0'), 13, tmp_path, device='cuda:1')
