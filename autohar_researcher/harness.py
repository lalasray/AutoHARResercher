from __future__ import annotations

import json
import math
import random
from pathlib import Path

from autohar_researcher.types import CandidateConfig, CandidateResult


class DummyHARHarness:
    """Small but real synthetic IMU HAR train/validation pipeline."""

    LABELS = ["walk", "sit", "stairs"]
    CHANNELS = 6

    def evaluate(
        self,
        round_id: int,
        agent_name: str,
        branch_name: str,
        worktree_path: str,
        config: CandidateConfig,
        component_summaries: dict[str, str] | None = None,
        artifact_paths: dict[str, str] | None = None,
    ) -> CandidateResult:
        experiment = self._load_experiment(Path(worktree_path))
        train_samples = self._build_dataset(split="train", config=config)
        val_samples = self._build_dataset(split="val", config=config)

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
        macro_f1 = self._macro_f1(predictions, val_labels)
        accuracy = self._accuracy(predictions, val_labels)
        train_accuracy = self._accuracy(
            [self._predict(model, features) for features in train_features],
            train_labels,
        )

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
            val_score=round(macro_f1, 4),
            summary=summary,
            score_breakdown={
                "macro_f1": round(macro_f1, 4),
                "val_accuracy": round(accuracy, 4),
                "train_accuracy": round(train_accuracy, 4),
                "train_samples": float(len(train_samples)),
                "val_samples": float(len(val_samples)),
            },
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

    def _build_dataset(
        self,
        split: str,
        config: CandidateConfig,
    ) -> list[tuple[list[list[float]], str]]:
        samples: list[tuple[list[list[float]], str]] = []
        per_label = 36 if split == "train" else 18
        seed_offset = 0 if split == "train" else 10_000
        for label_index, label in enumerate(self.LABELS):
            for sample_index in range(per_label):
                seed = seed_offset + label_index * 1_000 + sample_index * 17 + config.window_size
                rng = random.Random(seed)
                window = self._generate_window(label=label, window_size=config.window_size, rng=rng)
                window = self._preprocess(window, config.preprocessing)
                if split == "train":
                    window = self._augment(window, config.augmentation, rng=rng)
                samples.append((window, label))
        return samples

    def _generate_window(
        self,
        label: str,
        window_size: int,
        rng: random.Random,
    ) -> list[list[float]]:
        window: list[list[float]] = []
        for t in range(window_size):
            phase = (2.0 * math.pi * t) / max(window_size, 1)
            if label == "walk":
                sample = [
                    0.9 * math.sin(phase) + rng.uniform(-0.08, 0.08),
                    0.5 * math.cos(phase) + rng.uniform(-0.08, 0.08),
                    1.0 + 0.2 * math.sin(2 * phase) + rng.uniform(-0.04, 0.04),
                    0.3 * math.sin(phase) + rng.uniform(-0.05, 0.05),
                    0.1 * math.cos(phase) + rng.uniform(-0.05, 0.05),
                    0.2 * math.sin(2 * phase) + rng.uniform(-0.05, 0.05),
                ]
            elif label == "sit":
                sample = [
                    rng.uniform(-0.03, 0.03),
                    rng.uniform(-0.03, 0.03),
                    1.0 + rng.uniform(-0.02, 0.02),
                    rng.uniform(-0.02, 0.02),
                    rng.uniform(-0.02, 0.02),
                    rng.uniform(-0.02, 0.02),
                ]
            else:
                sample = [
                    1.1 * abs(math.sin(phase)) + rng.uniform(-0.10, 0.10),
                    0.6 * math.sin(phase + 0.7) + rng.uniform(-0.08, 0.08),
                    1.1 + 0.35 * math.cos(phase) + rng.uniform(-0.05, 0.05),
                    0.4 * abs(math.cos(phase)) + rng.uniform(-0.05, 0.05),
                    0.25 * math.sin(phase) + rng.uniform(-0.05, 0.05),
                    0.3 * math.cos(2 * phase) + rng.uniform(-0.05, 0.05),
                ]
            window.append(sample)
        return window

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
            return [[value + rng.uniform(-0.03, 0.03) for value in row] for row in augmented]
        if mode == "rotation":
            angle = math.pi / 12.0
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
                source_index = min(len(augmented) - 1, int(index * 0.9))
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
        scaled_lr = max(0.02, min(learning_rate * 800.0, 0.35))
        order = list(range(len(train_features)))

        for epoch in range(max(epochs * 12, 12)):
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
