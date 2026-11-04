import numpy as np
import pytest
import torch
from research.augmentations import validate_method
from research.torch_augmentations import augment
from research.preprocessing import gravity_reference, rotate_sensor_frame, causal_probability_mean


@pytest.mark.parametrize('device', ['cpu'] + (['cuda'] if torch.cuda.is_available() else []))
def test_gravity_tilt_is_physical_and_preserves_shared_triplet_rotation(device):
    raw = torch.tensor([4., 8., 2.], device=device).view(1, 1, 3).expand(3, 30, 3)
    raw = torch.cat([raw, 2 * raw], dim=2)
    class Normalizer:
        mean = np.array([1., 2., 3., 4., 5., 6.], dtype=np.float32)
        scale = np.array([2., 3., 4., 5., 6., 7.], dtype=np.float32)
    mean = torch.tensor(Normalizer.mean, device=device)
    scale = torch.tensor(Normalizer.scale, device=device)
    x = (raw - mean) / scale
    p = {'kind': 'gravity_tilt', 'params': {'min_degrees': 25., 'max_degrees': 25.}}
    result = augment(x, p, torch.Generator(device=device).manual_seed(13), normalizer=Normalizer)
    rotated = result * scale + mean
    torch.testing.assert_close(rotated[:, :, :3].norm(dim=2), raw[:, :, :3].norm(dim=2))
    torch.testing.assert_close(rotated[:, :, 3:], 2 * rotated[:, :, :3])
    direction = rotated[:, :, :3].mean(1)
    angle = torch.rad2deg(torch.acos(direction[:, 1].abs() / direction.norm(dim=1)))
    torch.testing.assert_close(angle, torch.full_like(angle, 25), atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize('params', [{'axis': 3}, {'axis': True}, {'min_degrees': 50, 'max_degrees': 20}, {'max_degrees': 91}])
def test_invalid_tilt_parameters(params):
    with pytest.raises(ValueError):
        validate_method({'kind': 'gravity_tilt', 'params': params})


def test_prefix_calibration_does_not_inspect_future_and_rotation_preserves_norms():
    prefix = np.tile([1., 9., 2.], (200, 1)).astype(np.float32)
    reference = gravity_reference(np.concatenate([prefix, np.full((10, 3), np.nan)]), 20)
    np.testing.assert_array_equal(reference, gravity_reference(prefix, 20))
    x = np.random.default_rng(13).normal(size=(3, 50, 6)).astype(np.float32)
    rotated = rotate_sensor_frame(x, reference)
    np.testing.assert_allclose(np.linalg.norm(x.reshape(3, 50, 2, 3), axis=3),
                               np.linalg.norm(rotated.reshape(3, 50, 2, 3), axis=3), rtol=1e-5)
    np.testing.assert_allclose(rotate_sensor_frame(prefix[None], reference).mean(1)[0],
                               [0, np.linalg.norm(prefix[0]), 0], atol=1e-4)


def test_probability_smoothing_is_prefix_invariant_and_groups_are_independent():
    p = np.array([[.99, .01]] * 4 + [[.01, .99]] * 6)
    groups = ['stream'] * len(p)
    baseline = causal_probability_mean(p, groups, history=6, reset_consecutive=3)
    changed = p.copy(); changed[5:] = [.8, .2]
    np.testing.assert_array_equal(baseline[:5], causal_probability_mean(changed, groups, 6, 3)[:5])
    assert baseline[6].argmax() == 1
    split = causal_probability_mean(p, ['first'] * 4 + ['second'] * 6, 6, 3)
    np.testing.assert_array_equal(split[4], p[4])
