import importlib
import importlib.util
import json
from pathlib import Path
import numpy as np
import pytest
import torch
from autohar_researcher.datasets import make_task
from autohar_researcher.harness import HARHarness
from autohar_researcher.types import Pipeline
from research.modules.architectures.presets import model_defaults
from research.modules.architectures.provenance import model_provenance

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('source,name,stride,channels', [
    ('usc', 'motion_statistics', 2, 6),
    ('original_model.py', 'motion_statistics', 1, 3),
    ('slope_model.py', 'motion_slope', 1, 3),
])
def test_matches_saved_local_architecture_logits_and_input_gradients(source, name, stride, channels):
    torch.set_num_threads(1)
    if source == 'usc':
        path = ROOT / 'reproduction/USC_HAD/selection/selected.json'
        if not path.exists():
            pytest.skip('Optional frozen USC-HAD reproduction snapshot is not installed')
        saved = json.loads(path.read_text())
        namespace = {}
        exec(saved['pipeline']['code']['research/components/model.py'], namespace)
    else:
        path = ROOT / 'reproduction/UniMiB_SHAR' / source
        if not path.exists():
            pytest.skip('Optional frozen UniMiB reproduction snapshot is not installed')
        spec = importlib.util.spec_from_file_location('saved_motion', path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        namespace = vars(module)
    params = dict(model_defaults(name), input_channels=channels, custom_params={'initial_stride': stride})
    original = namespace['build_model'](60 * channels, 5, params).eval()
    registered = importlib.import_module('research.modules.architectures.' + name).build_model(60 * channels, 5, params).eval()
    registered.load_state_dict(original.state_dict())
    x = torch.randn(2, 60, channels, requires_grad=True)
    other = x.detach().clone().requires_grad_()
    a, b = original(x), registered(other)
    torch.testing.assert_close(a, b, rtol=0, atol=0)
    a.sum().backward(); b.sum().backward()
    torch.testing.assert_close(x.grad, other.grad, rtol=1e-5, atol=1e-7)
    assert model_provenance(params)['status'] == 'local_experimental'


@pytest.mark.parametrize('name', ['motion_statistics', 'motion_slope'])
def test_standard_harness_can_train_and_reload(name, manifest, tmp_path):
    torch.set_num_threads(1)
    task = make_task(manifest, 'loso', 'uci_har', '0', 13)
    h = HARHarness(task, 13, tmp_path / name, 'cpu')
    p = Pipeline.from_dict({'data': {'window_seconds': .5},
                            'model': {'architecture': name, 'max_epochs': 1, 'batch_size': 4}})
    x = np.random.default_rng(13).normal(size=(8, 25, 6)).astype(np.float32)
    model, epoch = h._fit(p, x, np.arange(8) % len(task.labels), epochs=1)
    assert epoch == 1
    model.eval()
    restored = h._build_model(p).eval()
    restored.load_state_dict(model.state_dict())
    with torch.no_grad():
        torch.testing.assert_close(model(torch.tensor(x)), restored(torch.tensor(x)))
        assert torch.isfinite(model(torch.zeros_like(torch.tensor(x)))).all()
