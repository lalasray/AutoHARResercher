"""Tests for harness module."""

from __future__ import annotations

import pytest

from autohar_researcher.harness import DummyHARHarness
from autohar_researcher.types import CandidateConfig


class TestDummyHARHarness:
    def test_initialization(self):
        harness = DummyHARHarness()
        assert harness._dataset is None
        assert harness._torch is None or harness._torch is not None

    def test_runtime_backend_summary(self):
        harness = DummyHARHarness()
        summary = harness.runtime_backend_summary()
        assert isinstance(summary, str)
        assert len(summary) > 0

    def test_macro_f1_perfect(self):
        harness = DummyHARHarness()
        predictions = [0, 1, 2, 3, 4, 5] * 10
        labels = [0, 1, 2, 3, 4, 5] * 10
        score = harness._macro_f1(predictions, labels)
        assert score == 1.0

    def test_macro_f1_random(self):
        harness = DummyHARHarness()
        predictions = [0] * 60
        labels = list(range(6)) * 10
        score = harness._macro_f1(predictions, labels)
        assert 0 <= score <= 1

    def test_macro_f1_empty(self):
        harness = DummyHARHarness()
        score = harness._macro_f1([], [])
        assert score == 0.0

    def test_accuracy_perfect(self):
        harness = DummyHARHarness()
        predictions = [0, 1, 2] * 10
        labels = [0, 1, 2] * 10
        acc = harness._accuracy(predictions, labels)
        assert acc == 1.0

    def test_accuracy_zero(self):
        harness = DummyHARHarness()
        predictions = [0] * 10
        labels = [1] * 10
        acc = harness._accuracy(predictions, labels)
        assert acc == 0.0

    def test_confusion_matrix(self):
        harness = DummyHARHarness()
        predictions = [0, 1, 0, 1, 2]
        labels = [0, 1, 0, 2, 2]
        matrix = harness._confusion_matrix(predictions, labels)
        assert len(matrix) == len(harness.LABELS)
        assert matrix[0][0] == 2
        assert matrix[1][1] == 1
        assert matrix[2][2] == 2
        assert matrix[1][2] == 1

    def test_per_class_f1(self):
        harness = DummyHARHarness()
        predictions = [0, 1, 0, 1, 2, 2]
        labels = [0, 1, 0, 1, 2, 2]
        scores = harness._per_class_f1(predictions, labels)
        assert len(scores) == len(harness.LABELS)
        for label_name in harness.LABELS:
            assert label_name in scores
            assert 0 <= scores[label_name] <= 1

    def test_per_class_f1_perfect(self):
        harness = DummyHARHarness()
        predictions = list(range(6)) * 10
        labels = list(range(6)) * 10
        scores = harness._per_class_f1(predictions, labels)
        for label_name in harness.LABELS:
            assert scores[label_name] == 1.0

    def test_softmax(self):
        harness = DummyHARHarness()
        logits = [1.0, 2.0, 0.5]
        probs = harness._softmax(logits)
        assert len(probs) == 3
        assert abs(sum(probs) - 1.0) < 1e-6
        assert all(0 <= p <= 1 for p in probs)

    def test_dot_product(self):
        harness = DummyHARHarness()
        weights = [0.5, 0.3, 0.2, 0.1]
        features = [1.0, 2.0, 3.0]
        result = harness._dot(weights, features)
        expected = 0.5 * 1.0 + 0.3 * 2.0 + 0.2 * 3.0 + 0.1
        assert abs(result - expected) < 1e-6

    def test_parameter_count_linear(self):
        harness = DummyHARHarness()
        model = {
            "kind": "linear",
            "weights": [[0.1, 0.2, 0.3, 0.4] for _ in range(6)],
        }
        count = harness._parameter_count(model)
        assert count == 6 * 4


class TestHarnessPreprocessing:
    def test_normalize(self):
        harness = DummyHARHarness()
        window = [
            [1.0, 2.0, 3.0],
            [2.0, 4.0, 6.0],
            [3.0, 6.0, 9.0],
        ]
        normalized = harness._normalize(window)
        assert len(normalized) == len(window)
        for row in normalized:
            assert len(row) == len(window[0])

    def test_resample_nearest_same_size(self):
        harness = DummyHARHarness()
        window = [[1.0, 2.0, 3.0] for _ in range(128)]
        resampled = harness._resample_window(window, 128, {})
        assert len(resampled) == 128

    def test_resample_nearest_downsample(self):
        harness = DummyHARHarness()
        window = [[1.0, 2.0, 3.0] for _ in range(128)]
        resampled = harness._resample_window(window, 64, {"resample_mode": "nearest"})
        assert len(resampled) == 64

    def test_resample_linear_upsample(self):
        harness = DummyHARHarness()
        window = [[1.0, 2.0, 3.0] for _ in range(64)]
        resampled = harness._resample_window(window, 128, {"resample_mode": "linear"})
        assert len(resampled) == 128


class TestHarnessAugmentation:
    def test_augment_none(self):
        harness = DummyHARHarness()
        window = [[1.0, 2.0, 3.0] for _ in range(10)]
        augmented = harness._augment(
            window, "none", "medium", {}, rng=__import__("random").Random(42)
        )
        assert augmented == window

    def test_augment_jitter(self):
        harness = DummyHARHarness()
        window = [[1.0, 2.0, 3.0] for _ in range(10)]
        rng = __import__("random").Random(42)
        augmented = harness._augment(
            window, "jitter", "medium", {"jitter_std": 0.1}, rng=rng
        )
        assert len(augmented) == len(window)
        assert len(augmented[0]) == len(window[0])

    def test_augment_scaling(self):
        harness = DummyHARHarness()
        window = [[1.0, 2.0, 3.0] for _ in range(10)]
        rng = __import__("random").Random(42)
        augmented = harness._augment(
            window, "scaling", "medium", {"scale_span": 0.1}, rng=rng
        )
        assert len(augmented) == len(window)

    def test_augment_rotation(self):
        harness = DummyHARHarness()
        window = [[1.0, 2.0, 3.0, 0.1, 0.2, 0.3] for _ in range(10)]
        rng = __import__("random").Random(42)
        augmented = harness._augment(
            window, "rotation", "medium", {"angle_deg": 10.0}, rng=rng
        )
        assert len(augmented) == len(window)
        assert len(augmented[0]) == len(window[0])

    def test_augment_permutation(self):
        harness = DummyHARHarness()
        window = [[float(i), float(i), float(i)] for i in range(10)]
        rng = __import__("random").Random(42)
        augmented = harness._augment(
            window, "permutation", "medium", {"segment_count": 3}, rng=rng
        )
        assert len(augmented) == len(window)


class TestHarnessFeatures:
    def test_temporal_pool_avg(self):
        harness = DummyHARHarness()
        window = [[1.0, 2.0, 3.0] for _ in range(10)]
        features = harness._temporal_pool_features(window, "avg")
        assert len(features) == 3

    def test_temporal_pool_last(self):
        harness = DummyHARHarness()
        window = [[1.0, 2.0, 3.0] for _ in range(10)]
        features = harness._temporal_pool_features(window, "last")
        assert features == window[-1]

    def test_temporal_pool_concat(self):
        harness = DummyHARHarness()
        window = [[1.0, 2.0, 3.0] for _ in range(10)]
        features = harness._temporal_pool_features(window, "concat")
        assert len(features) == 6

    def test_attention_pool(self):
        harness = DummyHARHarness()
        window = [[1.0, 2.0, 3.0] for _ in range(10)]
        features = harness._attention_pool_features(window, 0.5)
        assert len(features) == 3

    def test_spectral_features(self):
        harness = DummyHARHarness()
        window = [[1.0, 2.0, 3.0] for _ in range(32)]
        features = harness._spectral_features(window, 4)
        assert len(features) == 12

    def test_multiscale_features(self):
        harness = DummyHARHarness()
        window = [[1.0, 2.0, 3.0] for _ in range(16)]
        features = harness._multiscale_features(window)
        assert len(features) > 0

    def test_interaction_features(self):
        harness = DummyHARHarness()
        window = [[1.0, 2.0, 3.0, 4.0] for _ in range(10)]
        features = harness._interaction_features(window)
        assert len(features) > 0


class TestHarnessTraining:
    def test_split_inner_validation(self):
        harness = DummyHARHarness()
        features = [[1.0, 2.0] for _ in range(60)]
        labels = [i % 6 for i in range(60)]
        train_feat, train_labels, stop_feat, stop_labels = (
            harness._split_inner_validation(features, labels)
        )
        assert len(train_feat) > 0
        assert len(stop_feat) > 0
        assert len(train_feat) + len(stop_feat) == len(features)
        assert len(train_labels) == len(train_feat)
        assert len(stop_labels) == len(stop_feat)


class TestCandidateConfig:
    def test_config_with_all_fields(self):
        config = CandidateConfig(
            model_name="deepconvlstm",
            preprocessing="normalized",
            augmentation="rotation",
            augmentation_strength="strong",
            learning_rate=0.001,
            window_size=128,
            model_params={"hidden_dim": 64},
            data_params={"normalize": True},
            augmentation_params={"angle_deg": 15.0},
            notes="test config",
        )
        assert config.model_params == {"hidden_dim": 64}
        assert config.data_params == {"normalize": True}
        assert config.augmentation_params == {"angle_deg": 15.0}
        assert config.notes == "test config"
