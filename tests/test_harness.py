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
    generator = SimpleNamespace(servers={'text': {}}, generate=lambda request, length, channels, seed, sampling_hz=50:
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


@pytest.mark.parametrize('device', ['cpu', pytest.param('cuda', marks=pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA unavailable'))])
def test_parallel_fits_match_serial_and_preserve_refit_evidence(manifest, tmp_path, device):
    from autohar_researcher.parallel import evaluate_many
    torch.set_num_threads(1)
    task = make_task(manifest, 'loso', 'uci_har', '0')
    serial = HARHarness(task, 13, tmp_path / 'serial', device=device)
    parallel = HARHarness(task, 13, tmp_path / 'parallel', device=device)
    pipelines = [small_pipeline(), small_pipeline()]
    pipelines[1].model['learning_rate'] = .002
    expected = [serial.evaluate(p) for p in pipelines]
    actual = list(evaluate_many(parallel, pipelines, 2, 1))
    assert actual == expected
    assert parallel._evaluations == serial._evaluations
    assert parallel._lr_schedules == serial._lr_schedules
    assert parallel._validation_diagnostics == serial._validation_diagnostics
    report = parallel.finalize(pipelines[0])
    assert report['refit_epochs'] == actual[0].best_epoch
    with pytest.raises(RuntimeError, match='closed'):
        list(evaluate_many(parallel, pipelines, 2, 1))


def test_model_code_can_edit_author_model_without_replacing_architecture(manifest, tmp_path):
    task = make_task(manifest, 'loso', 'uci_har', '0')
    harness = HARHarness(task, 13, tmp_path / 'code', device='cpu')
    pipeline = Pipeline.from_dict({'model': {'architecture': 'bilstmattention'}})
    pipeline.data['window_seconds'] = 1
    pipeline.code['research/components/model.py'] = '''def build_model(input_dim, num_classes, params):
    model = baseline_model(input_dim, num_classes, params)
    model.dropout.p = 0.3
    return model
'''
    model = harness._build_model(pipeline)
    assert model.__class__.__name__ == 'BiLSTMAttention'
    assert model.dropout.p == .3
    assert len(model.branches) == len(task.channels) // 3
    harness.preflight(pipeline)


def test_uncapped_monitor_stops_on_patience_beyond_configured_max(manifest, tmp_path, monkeypatch):
    task = make_task(manifest, 'loso', 'uci_har', '0')
    harness = HARHarness(task, 13, tmp_path / 'uncapped', device='cpu')
    pipeline = small_pipeline()
    pipeline.model.update(max_epochs=1, patience=2)
    x, y, normalizer = harness._training_arrays(pipeline, fit=True)
    # Constant scores improve only at epoch 1, then stop after two stale epochs.
    monkeypatch.setattr(harness, '_predict', lambda model, signals, batch_size: np.zeros(len(signals), dtype=int))
    _, best = harness._fit(pipeline, x, y, (x, y, np.array(['uci_har'] * len(y))),
                          normalizer=normalizer, uncapped=True, monitor_name='test_macro_f1')
    rows = [json.loads(line) for line in (harness.output / 'training-progress.jsonl').read_text().splitlines()]
    assert best == 1
    assert len(rows) == 3  # Goes past max_epochs=1, but respects patience.
    assert all(row['max_epochs'] is None and row['stage'] == 'test_selected' for row in rows)
    assert all('test_macro_f1' in row and 'validation_macro_f1' not in row for row in rows)


def test_custom_optimizer_loss_and_component_parameters(manifest, tmp_path):
    task = make_task(manifest, 'loso', 'uci_har', '0')
    harness = HARHarness(task, 13, tmp_path / 'hooks', device='cpu')
    p = small_pipeline()
    p.model['custom_params'] = {'learning_rate': .004}
    p.code['research/components/model.py'] = '''import torch
def build_optimizer(model, params):
    return torch.optim.SGD(model.parameters(), lr=component_params['learning_rate'])
def build_loss(params, class_weights):
    return torch.nn.CrossEntropyLoss(weight=class_weights)
'''
    optimizer = harness._optimizer(harness._build_model(p), p)
    assert isinstance(optimizer, torch.optim.SGD)
    assert optimizer.param_groups[0]['lr'] == .004
    assert 0 <= harness.evaluate(p).score <= 1


def test_class_aware_custom_augmentation(manifest, tmp_path):
    task = make_task(manifest, 'loso', 'uci_har', '0')
    harness = HARHarness(task, 13, tmp_path / 'aug-labels', device='cpu')
    p = small_pipeline()
    p.augmentation.update(kind='custom', probability=1., custom_params={'scale':.5})
    p.code['research/components/augmentation.py'] = '''def augment_labeled(x, labels, rng):
    return x * (1 - (labels == 0).to(x.dtype)[:, None, None] * component_params['scale'])
'''
    x = torch.ones(2, 50, 6)
    result = harness._augment(p, x, torch.Generator(), labels=torch.tensor([0,1]))
    torch.testing.assert_close(result[0], x[0] * .5)
    torch.testing.assert_close(result[1], x[1])


def fitted_pipeline():
    p = small_pipeline()
    p.data['normalization'] = 'none'
    p.code['research/components/data.py'] = '''import numpy as np
def fit_preprocessor(x_train, params):
    center = np.median(x_train, axis=(0, 1), keepdims=True)
    q = np.percentile(x_train, [25, 75], axis=(0, 1), keepdims=True)
    return {'center': center, 'scale': np.maximum(q[1]-q[0], 1e-6), 'windows': len(x_train)}
def transform(x, state):
    return (x-state['center'])/state['scale']
'''
    return p


def test_fitted_preprocessing_uses_training_only_and_refits_for_final(manifest, tmp_path, monkeypatch):
    from autohar_researcher.datasets import windows
    task = make_task(manifest, 'loso', 'uci_har', '0')
    harness = HARHarness(task, 13, tmp_path / 'fitted', device='cpu')
    p = fitted_pipeline()
    raw, _, _ = windows(task.train, task.labels, 1, 0)
    harness.preflight(p)
    result = harness.evaluate(p)
    state = harness._preprocessing_states[harness._key(p)]
    np.testing.assert_allclose(state['center'], np.median(raw, axis=(0, 1), keepdims=True))
    assert state['windows'] == len(raw)
    # Validation transformation never refits the learned statistics.
    center = state['center'].copy()
    harness._preprocess(p, np.full_like(raw[:2], 100000))
    np.testing.assert_array_equal(state['center'], center)
    assert (harness.output / 'preprocessing-state.pt').exists()
    harness.finalize(p)
    combined, _, _ = windows(task.train + task.validation, task.labels, 1, 0)
    final_state = harness._preprocessing_states[harness._key(p)]
    assert final_state['windows'] == len(combined)
    checkpoint = torch.load(harness.output / 'final-model.pt', weights_only=False)
    np.testing.assert_array_equal(checkpoint['preprocessing_state']['center'], final_state['center'])


def test_fitted_transform_cannot_mutate_stored_state(manifest, tmp_path):
    task = make_task(manifest, 'loso', 'uci_har', '0')
    h = HARHarness(task, 13, tmp_path / 'state-copy', device='cpu')
    p = fitted_pipeline()
    p.code['research/components/data.py'] += "\ndef transform(x, state):\n    state['center'][:] = 999\n    return x\n"
    h._training_arrays(p, fit=False)
    assert not np.all(h._preprocessing_states[h._key(p)]['center'] == 999)


def test_parallel_workers_return_fitted_state(manifest, tmp_path):
    from autohar_researcher.parallel import evaluate_many
    task = make_task(manifest, 'loso', 'uci_har', '0')
    h = HARHarness(task, 13, tmp_path / 'parallel-state', device='cpu')
    p = fitted_pipeline()
    results = list(evaluate_many(h, [p], 2, 1))
    assert not isinstance(results[0], Exception)
    assert h._preprocessing_states[h._key(p)]['windows'] > 0


def test_fitted_hook_requires_pair_and_fit_before_transform(manifest, tmp_path):
    task = make_task(manifest, 'loso', 'uci_har', '0')
    h = HARHarness(task, 13, tmp_path / 'pair', device='cpu')
    p = fitted_pipeline()
    with pytest.raises(ValueError, match='must be fitted'):
        h._preprocess(p, np.zeros((2,50,6), dtype=np.float32))
    p.code['research/components/data.py'] = 'def transform(x,state):\n    return x'
    with pytest.raises(ValueError, match='requires both'):
        h.preflight(p)
