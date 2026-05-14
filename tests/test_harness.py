"""Tests for harness module."""

from __future__ import annotations

import pytest

from autohar_researcher.harness import DummyHARHarness
from autohar_researcher.datasets import create_har_dataset, readily_downloadable_imu_datasets
from autohar_researcher.types import CandidateConfig


class TestDummyHARHarness:
    def test_initialization(self):
        harness = DummyHARHarness()
        assert harness._dataset is None

    def test_dataset_factory_supports_public_imu_datasets(self):
        names = [
            "wisdm",
            "uci_har",
            "utd_mhad",
            "hapt",
            "usc_had",
            "unimib_shar",
            "motion_sense",
            "real_life_har",
            "wisdm_19_phone",
            "wisdm_19_watch",
            "ku_har",
            "hang_time",
            "capture_24",
        ]
        statuses = readily_downloadable_imu_datasets()
        for name in names:
            dataset = create_har_dataset(name)
            assert dataset.metadata()["dataset_name"] == name
            assert statuses[name]["loader_status"] == "whar"

    def test_dataset_factory_supports_whar_display_aliases(self):
        aliases = {
            "UCI-HAR": "uci_har",
            "UTD-MHAD": "utd_mhad",
            "UniMiB-SHAR": "unimib_shar",
            "MotionSense": "motion_sense",
            "RealLifeHAR": "real_life_har",
            "WISDM-19-PHONE": "wisdm_19_phone",
            "WISDM-19-WATCH": "wisdm_19_watch",
            "KU-HAR": "ku_har",
            "Hang-Time": "hang_time",
            "CAPTURE-24": "capture_24",
        }
        for alias, expected in aliases.items():
            assert create_har_dataset(alias).metadata()["dataset_name"] == expected

    def test_cross_dataset_class_alignment_uses_common_classes(self):
        harness = DummyHARHarness()
        train_samples = [
            ([[1.0]], "Walking"),
            ([[2.0]], "Sitting"),
            ([[3.0]], "Cycling"),
        ]
        val_samples = [
            ([[4.0]], "walk"),
            ([[5.0]], "sit"),
            ([[6.0]], "stand"),
        ]

        aligned_train, aligned_val, labels = harness._align_dataset_classes(
            train_samples=train_samples,
            val_samples=val_samples,
            data_params={"target_classes": "all"},
        )

        assert labels == ["walk", "sit"]
        assert [label for _, label in aligned_train] == ["walk", "sit"]
        assert [label for _, label in aligned_val] == ["walk", "sit"]

    def test_cross_dataset_class_alignment_honors_target_classes(self):
        harness = DummyHARHarness()
        train_samples = [([[1.0]], "Walking"), ([[2.0]], "Sitting")]
        val_samples = [([[3.0]], "walk"), ([[4.0]], "sit")]

        aligned_train, aligned_val, labels = harness._align_dataset_classes(
            train_samples=train_samples,
            val_samples=val_samples,
            data_params={"target_classes": "Walking"},
        )

        assert labels == ["walk"]
        assert [label for _, label in aligned_train] == ["walk"]
        assert [label for _, label in aligned_val] == ["walk"]

    def test_resolve_dataset_plan_supports_multi_real_and_synthetic_sources(self):
        harness = DummyHARHarness()
        train_names, eval_name, synthetic_sources = harness._resolve_dataset_plan(
            {"dataset": {"name": "uci_har"}},
            {
                "dataset_policy": "real_synthetic_mix",
                "train_datasets": "wisdm,uci_har,hapt",
                "dataset_sources": "synthetic_imu_to_imu",
                "evaluation_dataset": "motion_sense",
                "synthetic_sources": "synthetic_text_to_imu,synthetic_imu_to_imu",
            },
        )

        assert train_names == ["wisdm", "uci_har", "hapt"]
        assert eval_name == "motion_sense"
        assert synthetic_sources == ["synthetic_text_to_imu", "synthetic_imu_to_imu"]

    def test_resolve_dataset_plan_uses_train_datasets(self):
        harness = DummyHARHarness()
        train_names, eval_name, synthetic_sources = harness._resolve_dataset_plan(
            {"dataset": {"name": "uci_har"}},
            {
                "dataset_policy": "cross_dataset",
                "train_datasets": "wisdm,uci_har,hapt",
                "evaluation_dataset": "usc_had",
            },
        )

        assert train_names == ["wisdm", "uci_har", "hapt"]
        assert eval_name == "usc_had"
        assert synthetic_sources == []

    def test_load_training_samples_supports_dataset_and_class_ratios(self):
        class FakeDataset:
            def __init__(self, label_prefix: str = "") -> None:
                self.label_prefix = label_prefix

            def ensure_downloaded(self):
                return None

            def load_split(self, split: str):
                assert split == "train"
                return [
                    ([[1.0]], "walk"),
                    ([[2.0]], "walk"),
                    ([[3.0]], "sit"),
                    ([[4.0]], "sit"),
                ]

            def metadata(self):
                return {"activity_labels": ["walk", "sit"]}

        harness = DummyHARHarness()
        harness._dataset_cache = {"wisdm": FakeDataset(), "uci_har": FakeDataset()}

        samples = harness._load_training_samples(
            ["wisdm", "uci_har"],
            {
                "per_dataset_real_ratio": {"wisdm": 0.5, "uci_har": 1.0},
                "per_class_real_ratio": {"walk": 0.5, "sit": 1.0},
                "per_dataset_class_real_ratio": {"uci_har": {"sit": 0.5}},
            },
        )

        labels = [label for _, label in samples]
        assert labels.count("walk") == 2
        assert labels.count("sit") == 2

    def test_synthetic_training_samples_add_rows(self):
        harness = DummyHARHarness()
        train_samples = [
            ([[1.0, 2.0, 3.0]], "walk"),
            ([[4.0, 5.0, 6.0]], "sit"),
        ]
        config = CandidateConfig(
            model_name="cnn",
            preprocessing="raw",
            augmentation="none",
            augmentation_strength="medium",
            learning_rate=0.001,
            window_size=64,
            data_params={"synthetic_ratio": 0.5},
            augmentation_params={"variation_amount": 0.1},
        )

        expanded = harness._add_synthetic_training_samples(
            train_samples,
            ["synthetic_imu_to_imu"],
            config,
        )

        assert len(expanded) == 3
        assert expanded[-1][1] in {"walk", "sit"}

    def test_synthetic_training_samples_honor_source_class_targets(self):
        harness = DummyHARHarness()
        train_samples = [
            ([[1.0, 2.0, 3.0]], "walk"),
            ([[4.0, 5.0, 6.0]], "sit"),
            ([[7.0, 8.0, 9.0]], "stand"),
        ]
        config = CandidateConfig(
            model_name="cnn",
            preprocessing="raw",
            augmentation="none",
            augmentation_strength="medium",
            learning_rate=0.001,
            window_size=64,
            data_params={
                "synthetic_ratio": 1.0,
                "synthetic_source_targets": {
                    "synthetic_text_to_imu": "sit",
                    "synthetic_video_to_imu": "walk",
                },
                "per_source_synthetic_ratio": {
                    "synthetic_text_to_imu": 1.0,
                    "synthetic_video_to_imu": 1.0,
                },
            },
            augmentation_params={"variation_amount": 0.1},
        )

        expanded = harness._add_synthetic_training_samples(
            train_samples,
            ["synthetic_text_to_imu", "synthetic_video_to_imu"],
            config,
        )

        synthetic_labels = [label for _, label in expanded[len(train_samples) :]]
        assert synthetic_labels == ["sit", "walk"]

    def test_synthetic_target_classes_apply_to_all_sources(self):
        harness = DummyHARHarness()
        train_samples = [
            ([[1.0, 2.0, 3.0]], "walk"),
            ([[4.0, 5.0, 6.0]], "sit"),
        ]
        config = CandidateConfig(
            model_name="cnn",
            preprocessing="raw",
            augmentation="none",
            augmentation_strength="medium",
            learning_rate=0.001,
            window_size=64,
            data_params={
                "synthetic_ratio": 1.0,
                "synthetic_target_classes": "sit",
            },
            augmentation_params={"variation_amount": 0.1},
        )

        expanded = harness._add_synthetic_training_samples(
            train_samples,
            ["synthetic_imu_to_imu"],
            config,
        )

        assert [label for _, label in expanded[len(train_samples) :]] == ["sit"]

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
        assert matrix[2][2] == 1
        assert matrix[2][1] == 1

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
