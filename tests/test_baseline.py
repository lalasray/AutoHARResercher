import json
from pathlib import Path

import numpy as np
import pytest
import torch

from autohar_researcher.datasets import make_task
from autohar_researcher.harness import HARHarness
from autohar_researcher.patches import validate_pipeline
from autohar_researcher.types import Pipeline
from research.modules.architectures.bilstmattention import build_model

ROOT = Path(__file__).resolve().parents[1]


def test_default_and_example_use_generic_cnn():
    config = Pipeline()
    assert json.loads((ROOT / 'examples/baseline.json').read_text()) == config.payload()
    assert config.model['architecture'] == 'cnn'
    validate_pipeline(config, {'uci_har'})


@pytest.mark.parametrize('channels,steps', [(3, 50), (6, 128), (9, 128), (6, 250)])
def test_reference_model_handles_fixed_sensor_layouts_and_search_windows(channels, steps):
    torch.set_num_threads(1)
    params = dict(Pipeline.from_dict({'model': {'architecture': 'bilstmattention'}}).model, input_channels=channels)
    model = build_model(steps * channels, 5, params)
    assert len(model.branches) == channels // 3
    x = np.random.default_rng(13).normal(size=(2, steps, channels)).astype('float32')
    logits = model(torch.from_numpy(x))
    assert logits.shape == (2, 5)
    optimizer = torch.optim.NAdam(model.parameters(), lr=.001)
    before = [w.detach().clone() for w in model.parameters()]
    loss = torch.nn.functional.cross_entropy(logits, torch.tensor([0, 1]))
    loss.backward()
    optimizer.step()
    assert torch.isfinite(loss)
    assert any(not torch.equal(a, b) for a, b in zip(before, model.parameters()))
    model.eval()
    assert torch.isfinite(model(torch.from_numpy(x[:1]))).all()


def test_author_optimizer_is_used_and_singleton_batches_work():
    config = Pipeline.from_dict({'model': {'architecture': 'bilstmattention'}})
    model = build_model(128 * 3, 5, dict(config.model, input_channels=3))
    optimizer = HARHarness._optimizer(model, config)
    assert isinstance(optimizer, torch.optim.NAdam)
    assert optimizer.param_groups[0]['lr'] == pytest.approx(.001)
    assert optimizer.param_groups[0]['betas'] == (.9, .999)
    assert optimizer.param_groups[0]['eps'] == 1e-7
    x = torch.randn(1, 128, 3)
    loss = torch.nn.functional.cross_entropy(model(x), torch.tensor([1]))
    loss.backward()
    optimizer.step()
    assert torch.isfinite(loss)


def test_selected_baseline_trains_without_test_access(manifest, tmp_path, monkeypatch):
    torch.set_num_threads(1)
    task = make_task(manifest, 'loso', 'uci_har', '0')
    test_paths = {r.path for r in task.test}
    original = np.load
    def source_only(path, *args, **kwargs):
        assert path not in test_paths
        return original(path, *args, **kwargs)
    monkeypatch.setattr(np, 'load', source_only)
    config = Pipeline()
    config.data['window_seconds'] = 1
    config.model['max_epochs'] = 1
    harness = HARHarness(task, 13, tmp_path / 'baseline', device='cpu')
    harness.preflight(config)
    result = harness.evaluate(config)
    assert 0 <= result.score <= 1
    assert result.best_epoch == 1
