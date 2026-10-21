import numpy as np
import pytest

from research.augmentations import CATALOG, MIXING, validate_policy
from research import torch_augmentations
import torch


def augment(x, policy, rng, labels=None, normalizer=None):
    generator = torch.Generator().manual_seed(int(rng.integers(2**31)))
    labels = None if labels is None else torch.as_tensor(labels)
    return torch_augmentations.augment(torch.as_tensor(x, dtype=torch.float32), policy,
                                      generator, labels, normalizer).numpy()


def mix_batch(x, labels, policy, rng, classes):
    generator = torch.Generator().manual_seed(int(rng.integers(2**31)))
    a, b = torch_augmentations.mix_batch(torch.as_tensor(x), torch.as_tensor(labels),
                                        policy, generator, classes)
    return a.numpy(), b.numpy()
from autohar_researcher.harness import Normalizer


@pytest.mark.parametrize('kind', sorted(set(CATALOG) - MIXING))
def test_signal_transforms_seeded_finite_and_nonmutating(kind):
    x = np.random.default_rng(3).normal(size=(4, 32, 6)).astype(np.float32)
    original = x.copy()
    policy = {'kind': kind}
    y = np.array([0, 0, 1, 1])
    a = augment(x, policy, np.random.default_rng(12), y)
    b = augment(x, policy, np.random.default_rng(12), y)
    assert a.shape == x.shape and a.dtype == np.float32
    assert np.isfinite(a).all()
    np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(x, original)
    np.testing.assert_array_equal(augment(x, dict(policy, probability=0), np.random.default_rng(12), y), x)


@pytest.mark.parametrize('kind', sorted(MIXING))
def test_mixed_labels_match_actual_signal_contribution(kind):
    y = np.arange(4)
    x = np.broadcast_to(np.eye(4, dtype=np.float32)[:, None, :], (4, 32, 4)).copy()
    a, targets = mix_batch(x, y, {'kind': kind}, np.random.default_rng(4), 4)
    np.testing.assert_allclose(a.mean(axis=1), targets, atol=1e-6)
    np.testing.assert_allclose(targets.sum(axis=1), 1)
    assert np.any((targets > 0) & (targets < 1))
    b, hard = mix_batch(x, y, {'kind': kind, 'probability': 0}, np.random.default_rng(4), 4)
    np.testing.assert_array_equal(b, x)
    np.testing.assert_array_equal(hard, np.eye(4))


def test_rotation_preserves_physical_norm_with_unequal_normalization():
    x = np.random.default_rng(3).normal(size=(4, 32, 6)).astype(np.float32)
    x *= np.arange(1, 7)
    x += np.arange(6) * 10
    normalizer = Normalizer('standard')
    normalizer.fit(x)
    rotated = augment(normalizer.transform(x), {'kind': 'rotation'}, np.random.default_rng(1), normalizer=normalizer)
    raw = rotated * normalizer.scale + normalizer.mean
    np.testing.assert_allclose(np.linalg.norm(raw.reshape(4, 32, 2, 3), axis=-1),
                               np.linalg.norm(x.reshape(4, 32, 2, 3), axis=-1), rtol=1e-6)


@pytest.mark.parametrize('kind', ['spawner', 'random_guided_warp'])
def test_class_aware_donors_cannot_cross_classes(kind):
    x = np.zeros((3, 24, 6), dtype=np.float32)
    x[2] = 100
    policy = {'kind': kind}
    if kind == 'spawner':
        policy['params'] = {'sigma': 0}
    result = augment(x, policy, np.random.default_rng(1), np.array([0, 0, 1]))
    np.testing.assert_array_equal(result, x)


def test_shared_monotone_time_warp():
    x = np.arange(32, dtype=np.float32)[None, :, None] * np.array([1, 2, 3])[None, None, :]
    result = augment(x, {'kind': 'time_warp'}, np.random.default_rng(1))
    assert (np.diff(result[0, :, 0]) >= 0).all()
    np.testing.assert_allclose(result[..., 1], 2 * result[..., 0])
    np.testing.assert_allclose(result[..., 2], 3 * result[..., 0], rtol=1e-6)


@pytest.mark.parametrize('policy', [
    {'kind': 'mixup', 'params': {'alpha': 0}},
    {'kind': 'compose', 'transforms': [{'kind': 'mixup'}]},
    {'kind': 'time_warp', 'params': {'sigma': float('nan')}},
    {'kind': 'window_slice', 'params': {'ratio': 0}},
])
def test_bad_policies_rejected(policy):
    with pytest.raises(ValueError):
        validate_policy(policy)


@pytest.mark.parametrize('kind', sorted(MIXING))
def test_harness_trains_with_soft_targets(manifest, tmp_path, kind):
    from autohar_researcher.datasets import make_task
    from autohar_researcher.harness import HARHarness
    from test_harness import small_pipeline
    p = small_pipeline()
    p.augmentation.update(kind='compose', transforms=[{'kind': 'time_warp'}],
                          mixing={'kind': kind})
    harness = HARHarness(make_task(manifest, 'loso', 'uci_har', '0'), 13, tmp_path / kind, device='cpu')
    harness.preflight(p)
    assert np.isfinite(harness.evaluate(p).score)
