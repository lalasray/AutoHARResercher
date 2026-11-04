import numpy as np
import pytest
import torch
from autohar_researcher.types import Pipeline
from autohar_researcher.datasets import make_task
from autohar_researcher.harness import HARHarness
from research.modules.architectures.feature_mlp import build_model
from research.modules.architectures.presets import model_defaults
from research.modules.architectures.provenance import model_provenance


@pytest.mark.parametrize('channels,hz,steps', [(3, 20, 100), (6, 50, 128), (1, 100, 50)])
def test_features_are_finite_and_normalization_is_checkpointed(channels, hz, steps):
    torch.set_num_threads(1)
    params = dict(model_defaults('feature_mlp'), input_channels=channels, sampling_hz=hz,
                  channel_names=('ax', 'ay', 'az', 'gx', 'gy', 'gz')[:channels])
    model = build_model(steps * channels, 3, params)
    x = torch.zeros(4, steps, channels)
    model.fit_feature_normalizer(x, 2)
    model.eval()
    before = model.feature_mean.clone()
    with torch.no_grad():
        assert torch.isfinite(model(x)).all()
        predictions = model(torch.ones_like(x) * 20)
    assert predictions.shape == (4, 3)
    assert torch.equal(before, model.feature_mean)
    assert all(hi <= hz / 2 for _, hi in model.frequency_bands)
    restored = build_model(steps * channels, 3, params).eval()
    restored.load_state_dict(model.state_dict())
    with torch.no_grad():
        torch.testing.assert_close(restored(torch.ones_like(x) * 20), predictions)
    assert model_provenance(params)['status'] == 'local_experimental'


def test_harness_fits_features_on_training_only(manifest, tmp_path, monkeypatch):
    torch.set_num_threads(1)
    task = make_task(manifest, 'loso', 'uci_har', '0', 13)
    harness = HARHarness(task, 13, tmp_path, 'cpu')
    pipeline = Pipeline.from_dict({'data': {'window_seconds': .5},
                                  'model': {'architecture': 'feature_mlp', 'max_epochs': 1,
                                            'batch_size': 4, 'hidden_dim': 16}})
    x = np.random.default_rng(13).normal(size=(8, 25, 6)).astype(np.float32)
    y = np.arange(8) % len(task.labels)
    reference = harness._build_model(pipeline)
    assert reference.sampling_hz == task.sampling_hz
    reference.fit_feature_normalizer(x, 4)
    validation = (np.full_like(x, 1000), y, np.array(['uci_har'] * len(y)))
    model, _ = harness._fit(pipeline, x, y, validation)
    assert model.statistics_fitted.item()
    torch.testing.assert_close(model.feature_mean, reference.feature_mean)
    torch.testing.assert_close(model.feature_scale, reference.feature_scale)
