from __future__ import annotations

import json
import math
import random
from pathlib import Path

from autohar_researcher.artifacts import ensure_round_dir, write_json_artifact
from autohar_researcher.datasets import UCIHARDataset
from autohar_researcher.types import CandidateConfig, CandidateResult


class DummyHARHarness:
    """Small but real HAR train/validation pipeline using UCI HAR."""

    LABELS = ["walk", "walk_upstairs", "walk_downstairs", "sit", "stand", "lay"]

    def __init__(self) -> None:
        self._dataset: UCIHARDataset | None = None
        self._evaluation_cache: dict[tuple[str, str, str, float, int], dict[str, object]] = {}

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
            config.augmentation_strength,
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
            confusion_matrix = self._confusion_matrix(predictions, val_labels)
            per_class_f1 = self._per_class_f1(predictions, val_labels)
            parameter_count = len(model) * len(model[0])
            feature_dim = len(train_features[0])
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
                "estimated_parameter_count": float(parameter_count),
                "feature_dim": float(feature_dim),
                "details": {
                    "labels": list(self.LABELS),
                    "confusion_matrix": confusion_matrix,
                    "per_class_f1": per_class_f1,
                },
            }
        cached = self._evaluation_cache[cache_key]
        summary = (
            f"{agent_name} proposed {config.model_name} with {config.preprocessing}, "
            f"{config.augmentation}:{config.augmentation_strength}, "
            f"lr={config.learning_rate}, window={config.window_size}"
        )
        details = dict(cached["details"])
        score_breakdown = {
            key: value
            for key, value in cached.items()
            if isinstance(value, float)
        }
        evaluation_artifact = self._write_evaluation_artifact(
            round_id=round_id,
            agent_name=agent_name,
            branch_name=branch_name,
            config=config,
            summary=summary,
            score_breakdown=score_breakdown,
            details=details,
        )
        merged_artifacts = dict(artifact_paths or {})
        merged_artifacts["evaluation_report"] = evaluation_artifact
        return CandidateResult(
            round_id=round_id,
            agent_name=agent_name,
            branch_name=branch_name,
            worktree_path=worktree_path,
            config=config,
            val_score=cached["macro_f1"],
            summary=summary,
            score_breakdown=score_breakdown,
            details=details,
            component_summaries=component_summaries or {},
            artifact_paths=merged_artifacts,
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
            processed = self._augment(
                processed,
                config.augmentation,
                config.augmentation_strength,
                rng=rng,
            )
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

    def _augment(
        self,
        window: list[list[float]],
        mode: str,
        strength: str,
        rng: random.Random,
    ) -> list[list[float]]:
        augmented = [row[:] for row in window]
        strength_scale = {
            "weak": 0.5,
            "medium": 1.0,
            "strong": 1.5,
        }.get(strength, 1.0)
        if mode == "none":
            return augmented
        if mode == "jitter":
            radius = 0.02 * strength_scale
            return [[value + rng.uniform(-radius, radius) for value in row] for row in augmented]
        if mode == "scaling":
            span = 0.1 * strength_scale
            scales = [rng.uniform(1.0 - span, 1.0 + span) for _ in range(len(augmented[0]))]
            return [[value * scales[index] for index, value in enumerate(row)] for row in augmented]
        if mode == "rotation":
            angle = (math.pi / 18.0) * strength_scale
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
            compression = max(0.7, 1.0 - (0.08 * strength_scale))
            for index in range(len(augmented)):
                source_index = min(len(augmented) - 1, int(index * compression))
                warped.append(augmented[source_index][:])
            return warped
        if mode == "permutation":
            if len(augmented) < 8:
                return augmented
            segment_count = min(6, max(3, int(round(2 + (2 * strength_scale)))))
            segment_size = max(len(augmented) // segment_count, 1)
            segments = [
                augmented[index : index + segment_size]
                for index in range(0, len(augmented), segment_size)
            ]
            rng.shuffle(segments)
            return [row[:] for segment in segments for row in segment]
        if mode == "drift":
            drift_radius = 0.015 * strength_scale
            drifts = [rng.uniform(-drift_radius, drift_radius) for _ in range(len(augmented[0]))]
            drifted: list[list[float]] = []
            for step, row in enumerate(augmented):
                alpha = step / max(len(augmented) - 1, 1)
                drifted.append(
                    [value + (alpha * drifts[index]) for index, value in enumerate(row)]
                )
            return drifted
        if mode == "magnitude_warp":
            anchors = [0.0, 0.33, 0.66, 1.0]
            span = 0.15 * strength_scale
            knots = [rng.uniform(1.0 - span, 1.0 + span) for _ in anchors]
            warped: list[list[float]] = []
            for step, row in enumerate(augmented):
                alpha = step / max(len(augmented) - 1, 1)
                left = max(index for index, anchor in enumerate(anchors) if anchor <= alpha)
                right = min(left + 1, len(anchors) - 1)
                if left == right:
                    scale = knots[left]
                else:
                    span = anchors[right] - anchors[left]
                    mix = 0.0 if span == 0 else (alpha - anchors[left]) / span
                    scale = ((1.0 - mix) * knots[left]) + (mix * knots[right])
                warped.append([value * scale for value in row])
            return warped
        if mode == "mixup":
            blend = min(0.45, 0.15 + (0.15 * strength_scale))
            mixed = []
            reversed_window = list(reversed(augmented))
            for left, right in zip(augmented, reversed_window):
                mixed.append([((1.0 - blend) * a) + (blend * b) for a, b in zip(left, right)])
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

    def _per_class_f1(self, predictions: list[int], labels: list[int]) -> dict[str, float]:
        scores: dict[str, float] = {}
        for class_index, label_name in enumerate(self.LABELS):
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
            score = 0.0 if precision + recall == 0 else (2.0 * precision * recall) / (precision + recall)
            scores[label_name] = round(score, 4)
        return scores

    def _confusion_matrix(self, predictions: list[int], labels: list[int]) -> list[list[int]]:
        matrix = [[0 for _ in self.LABELS] for _ in self.LABELS]
        for prediction, label in zip(predictions, labels):
            matrix[label][prediction] += 1
        return matrix

    def _write_evaluation_artifact(
        self,
        round_id: int,
        agent_name: str,
        branch_name: str,
        config: CandidateConfig,
        summary: str,
        score_breakdown: dict[str, float],
        details: dict[str, object],
    ) -> str:
        round_dir = ensure_round_dir(round_id)
        return write_json_artifact(
            round_dir / "results" / f"{agent_name}.json",
            {
                "round_id": round_id,
                "agent_name": agent_name,
                "branch_name": branch_name,
                "summary": summary,
                "config": {
                    "model_name": config.model_name,
                    "preprocessing": config.preprocessing,
                    "augmentation": config.augmentation,
                    "augmentation_strength": config.augmentation_strength,
                    "learning_rate": config.learning_rate,
                    "window_size": config.window_size,
                    "notes": config.notes,
                },
                "score_breakdown": score_breakdown,
                "details": details,
            },
        )
