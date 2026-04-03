from __future__ import annotations

import json
import math
import random
from pathlib import Path

from autohar_researcher.datasets import UCIHARDataset
from autohar_researcher.types import CandidateConfig, CandidateResult


class DummyHARHarness:
    """Small but real HAR train/validation pipeline using UCI HAR."""

    LABELS = ["walk", "walk_upstairs", "walk_downstairs", "sit", "stand", "lay"]

    def __init__(self) -> None:
        self._dataset: UCIHARDataset | None = None
        self._evaluation_cache: dict[tuple[str, str, str, float, int], dict[str, float]] = {}

    def evaluate(
        self,
        round_id: int,
        agent_name: str,
        branch_name: str,
        worktree_path: str,
        config: CandidateConfig,
        experiment: dict[str, object] | None = None,
        component_summaries: dict[str, str] | None = None,
        artifact_paths: dict[str, str] | None = None,
    ) -> CandidateResult:
        experiment = experiment or self._load_experiment(Path(worktree_path))
        cache_key = (
            config.model_name,
            config.preprocessing,
            config.augmentation,
            float(config.learning_rate),
            int(config.window_size),
        )
        if cache_key not in self._evaluation_cache:
            dataset = self._load_dataset(experiment)
            train_samples = [
                (self._prepare_window(window, config, split="train"), label)
                for window, label in dataset.load_split("train")
            ]
            val_samples = [
                (self._prepare_window(window, config, split="test"), label)
                for window, label in dataset.load_split("test")
            ]

            train_features = [self._extract_features(window, config) for window, _ in train_samples]
            train_labels = [self.LABELS.index(label) for _, label in train_samples]
            val_features = [self._extract_features(window, config) for window, _ in val_samples]
            val_labels = [self.LABELS.index(label) for _, label in val_samples]

            model = self._train_linear_model(
                train_features=train_features,
                train_labels=train_labels,
                learning_rate=config.learning_rate,
                epochs=int(experiment["training"]["epochs"]),
            )
            predictions = [self._predict(model, features) for features in val_features]
            self._evaluation_cache[cache_key] = {
                "macro_f1": round(self._macro_f1(predictions, val_labels), 4),
                "val_accuracy": round(self._accuracy(predictions, val_labels), 4),
                "train_accuracy": round(
                    self._accuracy(
                        [self._predict(model, features) for features in train_features],
                        train_labels,
                    ),
                    4,
                ),
                "train_samples": float(len(train_samples)),
                "val_samples": float(len(val_samples)),
            }
        cached = self._evaluation_cache[cache_key]

        summary = (
            f"{agent_name} proposed {config.model_name} with {config.preprocessing}, "
            f"{config.augmentation}, lr={config.learning_rate}, window={config.window_size}"
        )
        return CandidateResult(
            round_id=round_id,
            agent_name=agent_name,
            branch_name=branch_name,
            worktree_path=worktree_path,
            config=config,
            val_score=cached["macro_f1"],
            summary=summary,
            score_breakdown=dict(cached),
            component_summaries=component_summaries or {},
            artifact_paths=artifact_paths or {},
        )

    def _load_experiment(self, worktree_path: Path) -> dict[str, object]:
        experiment_path = worktree_path / "research" / "experiment.json"
        if not experiment_path.exists():
            return {
                "training": {"epochs": 5},
            }
        payload = json.loads(experiment_path.read_text(encoding="utf-8"))
        if "experiment" in payload:
            return payload["experiment"]
        return payload

    def _load_dataset(self, experiment: dict[str, object]) -> UCIHARDataset:
        dataset_name = str(experiment["dataset"]["name"])
        if dataset_name == "dummy_imu_har":
            dataset_name = "uci_har"
        if dataset_name != "uci_har":
            raise ValueError(f"Unsupported dataset {dataset_name!r}")
        if self._dataset is None:
            self._dataset = UCIHARDataset()
            self._dataset.ensure_downloaded()
        return self._dataset

    def _prepare_window(
        self,
        window: list[list[float]],
        config: CandidateConfig,
        split: str,
    ) -> list[list[float]]:
        processed = self._select_channels(window, config.preprocessing)
        processed = self._resample_window(processed, config.window_size)
        processed = self._preprocess(processed, config.preprocessing)
        if split == "train":
            rng = random.Random(config.window_size + len(window))
            processed = self._augment(processed, config.augmentation, rng=rng)
        return processed

    def _select_channels(self, window: list[list[float]], mode: str) -> list[list[float]]:
        selected: list[list[float]] = []
        for row in window:
            total_acc = row[:3]
            body_acc = row[3:6]
            body_gyro = row[6:9]
            if mode == "normalized+gravity_filter":
                selected.append(body_acc + body_gyro)
            else:
                selected.append(total_acc + body_gyro)
        return selected

    def _resample_window(self, window: list[list[float]], target_size: int) -> list[list[float]]:
        if len(window) == target_size:
            return [row[:] for row in window]
        resampled = []
        for idx in range(target_size):
            source = int(idx * (len(window) - 1) / max(target_size - 1, 1))
            resampled.append(window[source][:])
        return resampled

    def _preprocess(self, window: list[list[float]], mode: str) -> list[list[float]]:
        processed = [row[:] for row in window]
        if mode in {"normalized", "normalized+gravity_filter", "normalized+magnitude_channels"}:
            processed = self._normalize(processed)
        if mode == "normalized+gravity_filter":
            accel_means = [
                sum(row[channel] for row in processed) / len(processed)
                for channel in range(3)
            ]
            for row in processed:
                for channel in range(3):
                    row[channel] -= accel_means[channel]
        if mode == "normalized+magnitude_channels":
            extended: list[list[float]] = []
            for row in processed:
                accel_mag = math.sqrt(sum(value * value for value in row[:3]))
                gyro_mag = math.sqrt(sum(value * value for value in row[3:6]))
                extended.append(row + [accel_mag, gyro_mag])
            processed = extended
        return processed

    def _normalize(self, window: list[list[float]]) -> list[list[float]]:
        channels = len(window[0])
        means = [sum(row[i] for row in window) / len(window) for i in range(channels)]
        stds = []
        for i in range(channels):
            variance = sum((row[i] - means[i]) ** 2 for row in window) / len(window)
            stds.append(math.sqrt(variance) + 1e-6)
        normalized: list[list[float]] = []
        for row in window:
            normalized.append([(row[i] - means[i]) / stds[i] for i in range(channels)])
        return normalized

    def _augment(self, window: list[list[float]], mode: str, rng: random.Random) -> list[list[float]]:
        augmented = [row[:] for row in window]
        if mode == "none":
            return augmented
        if mode == "jitter":
            return [[value + rng.uniform(-0.02, 0.02) for value in row] for row in augmented]
        if mode == "rotation":
            angle = math.pi / 18.0
            cos_a = math.cos(angle)
            sin_a = math.sin(angle)
            rotated: list[list[float]] = []
            for row in augmented:
                ax, ay, az, gx, gy, gz = row[:6]
                rotated.append(
                    [
                        cos_a * ax - sin_a * ay,
                        sin_a * ax + cos_a * ay,
                        az,
                        cos_a * gx - sin_a * gy,
                        sin_a * gx + cos_a * gy,
                        gz,
                        *row[6:],
                    ]
                )
            return rotated
        if mode == "time_warp":
            warped = []
            for index in range(len(augmented)):
                source_index = min(len(augmented) - 1, int(index * 0.92))
                warped.append(augmented[source_index][:])
            return warped
        if mode == "mixup":
            mixed = []
            reversed_window = list(reversed(augmented))
            for left, right in zip(augmented, reversed_window):
                mixed.append([(0.7 * a) + (0.3 * b) for a, b in zip(left, right)])
            return mixed
        return augmented

    def _extract_features(self, window: list[list[float]], config: CandidateConfig) -> list[float]:
        channels = len(window[0])
        means = [sum(row[i] for row in window) / len(window) for i in range(channels)]
        stds = []
        energies = []
        diffs = []
        for i in range(channels):
            variance = sum((row[i] - means[i]) ** 2 for row in window) / len(window)
            stds.append(math.sqrt(variance))
            energies.append(sum(row[i] * row[i] for row in window) / len(window))
            diffs.append(
                sum(abs(window[idx][i] - window[idx - 1][i]) for idx in range(1, len(window)))
                / max(len(window) - 1, 1)
            )

        if config.model_name == "cnn":
            return means + stds + energies
        if config.model_name == "gru":
            return means + diffs + [window[-1][i] - window[0][i] for i in range(channels)]
        if config.model_name == "tiny_transformer":
            midpoint = len(window) // 2
            first_half = [sum(window[i][c] for i in range(midpoint)) / max(midpoint, 1) for c in range(channels)]
            second_half = [
                sum(window[i][c] for i in range(midpoint, len(window))) / max(len(window) - midpoint, 1)
                for c in range(channels)
            ]
            return means + stds + [second_half[i] - first_half[i] for i in range(channels)]
        return means + stds + energies + diffs

    def _train_linear_model(
        self,
        train_features: list[list[float]],
        train_labels: list[int],
        learning_rate: float,
        epochs: int,
    ) -> list[list[float]]:
        feature_dim = len(train_features[0])
        num_classes = len(self.LABELS)
        weights = [[0.0 for _ in range(feature_dim + 1)] for _ in range(num_classes)]
        scaled_lr = max(0.01, min(learning_rate * 500.0, 0.18))
        order = list(range(len(train_features)))

        for epoch in range(max(epochs * 4, 8)):
            random.Random(epochs + epoch).shuffle(order)
            for sample_index in order:
                features = train_features[sample_index]
                label = train_labels[sample_index]
                logits = [self._dot(weight, features) for weight in weights]
                probs = self._softmax(logits)
                for class_index in range(num_classes):
                    error = probs[class_index] - (1.0 if class_index == label else 0.0)
                    for feature_index, value in enumerate(features):
                        weights[class_index][feature_index] -= scaled_lr * error * value
                    weights[class_index][-1] -= scaled_lr * error
        return weights

    def _predict(self, weights: list[list[float]], features: list[float]) -> int:
        logits = [self._dot(weight, features) for weight in weights]
        return max(range(len(logits)), key=lambda index: logits[index])

    def _dot(self, weight: list[float], features: list[float]) -> float:
        return sum(coef * value for coef, value in zip(weight[:-1], features)) + weight[-1]

    def _softmax(self, logits: list[float]) -> list[float]:
        max_logit = max(logits)
        exps = [math.exp(logit - max_logit) for logit in logits]
        total = sum(exps)
        return [value / total for value in exps]

    def _accuracy(self, predictions: list[int], labels: list[int]) -> float:
        correct = sum(int(prediction == label) for prediction, label in zip(predictions, labels))
        return correct / max(len(labels), 1)

    def _macro_f1(self, predictions: list[int], labels: list[int]) -> float:
        scores = []
        for class_index in range(len(self.LABELS)):
            tp = sum(
                int(prediction == class_index and label == class_index)
                for prediction, label in zip(predictions, labels)
            )
            fp = sum(
                int(prediction == class_index and label != class_index)
                for prediction, label in zip(predictions, labels)
            )
            fn = sum(
                int(prediction != class_index and label == class_index)
                for prediction, label in zip(predictions, labels)
            )
            precision = tp / max(tp + fp, 1)
            recall = tp / max(tp + fn, 1)
            if precision + recall == 0:
                scores.append(0.0)
            else:
                scores.append((2.0 * precision * recall) / (precision + recall))
        return sum(scores) / len(scores)
