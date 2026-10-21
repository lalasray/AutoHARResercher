import numpy as np
import pytest
import torch

from research.augmentations import validate_policy, policy_provenance
from research.torch_augmentations import augment, mix_batch

DEVICES = ['cpu'] + (['cuda'] if torch.cuda.is_available() else [])
NAMES = ['walking', 'sitting', 'running']


@pytest.mark.parametrize('device', DEVICES)
def test_independent_methods_strength_probability_and_global_fallback(device):
    x = torch.ones(6, 32, 6, device=device)
    y = torch.tensor([0, 1, 2, 0, 1, 2], device=device)
    policy = {'kind': 'scaling', 'params': {'sigma': .2}, 'per_class': {
        'walking': {'kind': 'compose', 'transforms': [
            {'kind': 'jitter', 'params': {'sigma': 2}, 'probability': 0},
            {'kind': 'time_mask', 'params': {'ratio': 1}}]},
        'sitting': {'kind': 'none'}}}
    def run():
        return augment(x, policy, torch.Generator(device=device).manual_seed(13), y, class_names=NAMES)
    result = run()
    torch.testing.assert_close(result, run(), rtol=0, atol=0)
    torch.testing.assert_close(result[y == 1], x[y == 1])
    assert ((result[y == 0] == 0) | (result[y == 0] == 1)).all()
    assert (result[y == 0] == 0).any()
    assert not torch.equal(result[y == 2], x[y == 2])
    assert result.device == x.device
    torch.testing.assert_close(x, torch.ones_like(x))
    _, hard = mix_batch(result, y, policy, torch.Generator(device=device).manual_seed(13), 3, NAMES)
    assert hard is y


@pytest.mark.parametrize('device', DEVICES)
@pytest.mark.parametrize('kind', ['mixup', 'temporal_cutmix'])
def test_class_mixing_preserves_other_recipients_and_correct_soft_labels(device, kind):
    y = torch.tensor([0, 1, 2, 0, 1, 2], device=device)
    x = torch.nn.functional.one_hot(y, 3).float()[:, None].expand(-1, 64, -1).clone()
    policy = {'kind': 'none', 'per_class': {
        'walking': {'kind': kind, 'params': {'alpha': 1}},
        'sitting': {'kind': 'none'}}}
    output, target = mix_batch(x, y, policy, torch.Generator(device=device).manual_seed(3), 3, NAMES)
    torch.testing.assert_close(output.mean(1), target)
    torch.testing.assert_close(output[y != 0], x[y != 0])
    assert not torch.equal(output[y == 0], x[y == 0])
    assert output.device == x.device and target.device == x.device
    torch.testing.assert_close(target.sum(1), torch.ones(len(y), device=device))


@pytest.mark.parametrize('override', [
    {'kind': 'custom'}, {'kind': 'none', 'per_class': {}},
    {'kind': 'jitter', 'params': {'sigma': -1}},
    {'kind': 'none', 'synthetic': []}, {'params': {'sigma': .2}},
])
def test_invalid_class_policies_rejected(override):
    with pytest.raises(ValueError):
        validate_policy({'kind': 'none', 'per_class': {'walking': override}})


def test_unknown_class_rejected_before_model_build(manifest, tmp_path, monkeypatch):
    from autohar_researcher.harness import HARHarness
    from autohar_researcher.datasets import make_task
    from test_harness import small_pipeline
    p = small_pipeline()
    p.augmentation['per_class'] = {'typo': {'kind': 'jitter'}}
    harness = HARHarness(make_task(manifest, 'loso', 'uci_har', '0'), 13, tmp_path, device='cpu')
    monkeypatch.setattr(harness, '_build_model', lambda p: pytest.fail('Should reject before building model'))
    with pytest.raises(ValueError, match='fixed label space'):
        harness.preflight(p)


@pytest.mark.parametrize('device', DEVICES)
def test_training_only_and_provenance(manifest, tmp_path, monkeypatch, device):
    from autohar_researcher.harness import HARHarness
    from autohar_researcher.datasets import make_task
    from test_harness import small_pipeline
    task = make_task(manifest, 'loso', 'uci_har', '0')
    p = small_pipeline()
    label = task.labels[0]
    p.augmentation['per_class'] = {label: {
        'kind': 'jitter', 'params': {'sigma': .01},
        'mixing': {'kind': 'mixup', 'probability': .5}}}
    harness = HARHarness(task, 13, tmp_path, device=device)
    harness.preflight(p)
    original_load = np.load
    test_paths = {r.path for r in task.test}
    def load(path, *args, **kwargs):
        assert path not in test_paths
        return original_load(path, *args, **kwargs)
    monkeypatch.setattr(np, 'load', load)
    calls = []
    original_augment = harness._augment
    def checked(*args, **kwargs):
        calls.append(1)
        return original_augment(*args, **kwargs)
    monkeypatch.setattr(harness, '_augment', checked)
    assert np.isfinite(harness.evaluate(p).score)
    assert calls
    monkeypatch.setattr(harness, '_augment', lambda *a, **kw: pytest.fail('Inference must not augment'))
    harness._predict(harness._build_model(p), np.zeros((2, 50, len(task.channels)), dtype=np.float32), 2)
    provenance = policy_provenance(p.augmentation)['per_class'][label]
    assert provenance['signal_steps'][0]['params']['sigma'] == .01
    assert provenance['mixing']['kind'] == 'mixup'
    assert provenance['signal_steps'][0]['source'].startswith('https://')
