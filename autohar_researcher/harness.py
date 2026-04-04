from __future__ import annotations

import json
import math
import random
from pathlib import Path

try:
    import torch
except Exception:  # pragma: no cover - optional dependency path
    torch = None

from autohar_researcher.artifacts import ensure_round_dir, write_json_artifact
from autohar_researcher.datasets import UCIHARDataset
from autohar_researcher.types import CandidateConfig, CandidateResult


class DummyHARHarness:
    """Small but real HAR train/validation pipeline using UCI HAR."""

    LABELS = ["walk", "walk_upstairs", "walk_downstairs", "sit", "stand", "lay"]

    def __init__(self) -> None:
        self._dataset: UCIHARDataset | None = None
        self._evaluation_cache: dict[tuple[str, str, str, float, int], dict[str, object]] = {}
        self._torch = torch

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
            training = self._training_config(experiment)
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
            inner_train_features, inner_train_labels, stop_features, stop_labels = self._split_inner_validation(
                train_features,
                train_labels,
            )

            model, training_report = self._train_model(
                model_name=config.model_name,
                train_features=inner_train_features,
                train_labels=inner_train_labels,
                stop_features=stop_features,
                stop_labels=stop_labels,
                learning_rate=config.learning_rate,
                epochs=training["epochs"],
                patience=training["early_stopping_patience"],
                min_epochs=training["min_epochs"],
                device_preference=training["device"],
            )
            predictions = [self._predict(model, features) for features in val_features]
            confusion_matrix = self._confusion_matrix(predictions, val_labels)
            per_class_f1 = self._per_class_f1(predictions, val_labels)
            parameter_count = float(self._parameter_count(model))
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
                "epochs_trained": float(training_report["epochs_trained"]),
                "best_epoch": float(training_report["best_epoch"]),
                "checkpoint_payload": self._checkpoint_payload(
                    model=model,
                    config=config,
                    training_report=training_report,
                ),
                "details": {
                    "labels": list(self.LABELS),
                    "confusion_matrix": confusion_matrix,
                    "per_class_f1": per_class_f1,
                    "training_report": training_report,
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
        checkpoint_artifact = self._write_checkpoint_artifact(
            round_id=round_id,
            agent_name=agent_name,
            branch_name=branch_name,
            checkpoint_payload=cached["checkpoint_payload"],
        )
        merged_artifacts = dict(artifact_paths or {})
        merged_artifacts["evaluation_report"] = evaluation_artifact
        merged_artifacts["best_checkpoint"] = checkpoint_artifact
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

    def _training_config(self, experiment: dict[str, object]) -> dict[str, int | str]:
        training = experiment.get("training", {})
        return {
            "epochs": int(training.get("epochs", 12)),
            "min_epochs": int(training.get("min_epochs", 4)),
            "early_stopping_patience": int(training.get("early_stopping_patience", 3)),
            "device": str(training.get("device", "auto")),
        }

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

        multiscale = self._multiscale_features(window)
        saliency = self._saliency_features(window)
        interactions = self._interaction_features(window)

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
        if config.model_name == "deepconvlstm":
            return means + stds + energies + diffs + multiscale + interactions
        if config.model_name == "hart":
            return means + stds + saliency + multiscale + interactions
        if config.model_name == "tinierhar":
            return means + stds + saliency + interactions
        return means + stds + energies + diffs

    def _train_model(
        self,
        model_name: str,
        train_features: list[list[float]],
        train_labels: list[int],
        stop_features: list[list[float]],
        stop_labels: list[int],
        learning_rate: float,
        epochs: int,
        patience: int,
        min_epochs: int,
        device_preference: str,
    ) -> tuple[dict[str, object], dict[str, object]]:
        if self._torch is not None:
            return self._train_torch_model(
                model_name=model_name,
                train_features=train_features,
                train_labels=train_labels,
                stop_features=stop_features,
                stop_labels=stop_labels,
                learning_rate=learning_rate,
                epochs=epochs,
                patience=patience,
                min_epochs=min_epochs,
                device_preference=device_preference,
            )
        if model_name in {"deepconvlstm", "hart", "tinierhar"}:
            hidden_dim = 48 if model_name in {"deepconvlstm", "hart"} else 24
            return self._train_mlp_model(
                train_features=train_features,
                train_labels=train_labels,
                stop_features=stop_features,
                stop_labels=stop_labels,
                learning_rate=learning_rate,
                epochs=epochs,
                hidden_dim=hidden_dim,
                patience=patience,
                min_epochs=min_epochs,
            )
        weights = self._train_linear_model(
            train_features=train_features,
            train_labels=train_labels,
            stop_features=stop_features,
            stop_labels=stop_labels,
            learning_rate=learning_rate,
            epochs=epochs,
            patience=patience,
            min_epochs=min_epochs,
        )
        return weights

    def _train_linear_model(
        self,
        train_features: list[list[float]],
        train_labels: list[int],
        stop_features: list[list[float]],
        stop_labels: list[int],
        learning_rate: float,
        epochs: int,
        patience: int,
        min_epochs: int,
    ) -> tuple[dict[str, object], dict[str, object]]:
        feature_dim = len(train_features[0])
        num_classes = len(self.LABELS)
        weights = [[0.0 for _ in range(feature_dim + 1)] for _ in range(num_classes)]
        scaled_lr = max(0.01, min(learning_rate * 500.0, 0.18))
        order = list(range(len(train_features)))
        best_weights = [row[:] for row in weights]
        best_score = float("-inf")
        best_epoch = 0
        stale_epochs = 0

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
            stop_predictions = [self._predict({"kind": "linear", "weights": weights}, features) for features in stop_features]
            stop_score = self._macro_f1(stop_predictions, stop_labels)
            if stop_score > best_score:
                best_score = stop_score
                best_epoch = epoch + 1
                stale_epochs = 0
                best_weights = [row[:] for row in weights]
            else:
                stale_epochs += 1
            if (epoch + 1) >= min_epochs and stale_epochs >= patience:
                break
        return (
            {"kind": "linear", "weights": best_weights},
            {
                "device": "cpu",
                "epochs_trained": epoch + 1,
                "best_epoch": best_epoch,
                "best_stop_macro_f1": round(best_score, 4),
                "stopped_early": stale_epochs >= patience,
            },
        )

    def _train_mlp_model(
        self,
        train_features: list[list[float]],
        train_labels: list[int],
        stop_features: list[list[float]],
        stop_labels: list[int],
        learning_rate: float,
        epochs: int,
        hidden_dim: int,
        patience: int,
        min_epochs: int,
    ) -> tuple[dict[str, object], dict[str, object]]:
        feature_dim = len(train_features[0])
        num_classes = len(self.LABELS)
        rng = random.Random(feature_dim + hidden_dim + epochs)
        hidden = [
            [rng.uniform(-0.05, 0.05) for _ in range(feature_dim + 1)]
            for _ in range(hidden_dim)
        ]
        output = [
            [rng.uniform(-0.05, 0.05) for _ in range(hidden_dim + 1)]
            for _ in range(num_classes)
        ]
        scaled_lr = max(0.003, min(learning_rate * 60.0, 0.03))
        order = list(range(len(train_features)))
        best_hidden = [row[:] for row in hidden]
        best_output = [row[:] for row in output]
        best_score = float("-inf")
        best_epoch = 0
        stale_epochs = 0
        for epoch in range(max(epochs * 3, 6)):
            random.Random(hidden_dim + epoch).shuffle(order)
            for sample_index in order:
                features = train_features[sample_index]
                label = train_labels[sample_index]
                hidden_linear = [self._dot(weight, features) for weight in hidden]
                hidden_state = [math.tanh(value) for value in hidden_linear]
                logits = [self._dot(weight, hidden_state) for weight in output]
                probs = self._softmax(logits)
                output_errors = [
                    probs[class_index] - (1.0 if class_index == label else 0.0)
                    for class_index in range(num_classes)
                ]
                hidden_errors = []
                for hidden_index in range(hidden_dim):
                    downstream = sum(
                        output[class_index][hidden_index] * output_errors[class_index]
                        for class_index in range(num_classes)
                    )
                    hidden_errors.append((1.0 - (hidden_state[hidden_index] ** 2)) * downstream)
                for class_index in range(num_classes):
                    for hidden_index, value in enumerate(hidden_state):
                        output[class_index][hidden_index] -= scaled_lr * output_errors[class_index] * value
                    output[class_index][-1] -= scaled_lr * output_errors[class_index]
                for hidden_index in range(hidden_dim):
                    for feature_index, value in enumerate(features):
                        hidden[hidden_index][feature_index] -= scaled_lr * hidden_errors[hidden_index] * value
                    hidden[hidden_index][-1] -= scaled_lr * hidden_errors[hidden_index]
            stop_model = {"kind": "mlp", "hidden": hidden, "output": output}
            stop_predictions = [self._predict(stop_model, features) for features in stop_features]
            stop_score = self._macro_f1(stop_predictions, stop_labels)
            if stop_score > best_score:
                best_score = stop_score
                best_epoch = epoch + 1
                stale_epochs = 0
                best_hidden = [row[:] for row in hidden]
                best_output = [row[:] for row in output]
            else:
                stale_epochs += 1
            if (epoch + 1) >= min_epochs and stale_epochs >= patience:
                break
        return (
            {"kind": "mlp", "hidden": best_hidden, "output": best_output},
            {
                "device": "cpu",
                "epochs_trained": epoch + 1,
                "best_epoch": best_epoch,
                "best_stop_macro_f1": round(best_score, 4),
                "stopped_early": stale_epochs >= patience,
            },
        )

    def _train_torch_model(
        self,
        model_name: str,
        train_features: list[list[float]],
        train_labels: list[int],
        stop_features: list[list[float]],
        stop_labels: list[int],
        learning_rate: float,
        epochs: int,
        patience: int,
        min_epochs: int,
        device_preference: str,
    ) -> tuple[dict[str, object], dict[str, object]]:
        assert self._torch is not None
        feature_dim = len(train_features[0])
        num_classes = len(self.LABELS)
        device = self._resolve_device(device_preference)
        if model_name in {"deepconvlstm", "hart", "tinierhar"}:
            hidden_dim = 64 if model_name in {"deepconvlstm", "hart"} else 32
            module = self._torch.nn.Sequential(
                self._torch.nn.Linear(feature_dim, hidden_dim),
                self._torch.nn.ReLU(),
                self._torch.nn.Linear(hidden_dim, num_classes),
            )
        else:
            module = self._torch.nn.Linear(feature_dim, num_classes)
        module = module.to(device)
        optimizer = self._torch.optim.Adam(module.parameters(), lr=max(learning_rate, 1e-4))
        criterion = self._torch.nn.CrossEntropyLoss()
        train_x = self._torch.tensor(train_features, dtype=self._torch.float32, device=device)
        train_y = self._torch.tensor(train_labels, dtype=self._torch.long, device=device)
        stop_x = self._torch.tensor(stop_features, dtype=self._torch.float32, device=device)
        batch_size = min(128, len(train_features))
        best_state = {key: value.detach().cpu().clone() for key, value in module.state_dict().items()}
        best_score = float("-inf")
        best_epoch = 0
        stale_epochs = 0
        for epoch in range(epochs):
            module.train()
            permutation = self._torch.randperm(train_x.shape[0], device=device)
            for start in range(0, train_x.shape[0], batch_size):
                indices = permutation[start : start + batch_size]
                logits = module(train_x[indices])
                loss = criterion(logits, train_y[indices])
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            module.eval()
            with self._torch.no_grad():
                stop_logits = module(stop_x)
                stop_predictions = stop_logits.argmax(dim=1).detach().cpu().tolist()
            stop_score = self._macro_f1(stop_predictions, stop_labels)
            if stop_score > best_score:
                best_score = stop_score
                best_epoch = epoch + 1
                stale_epochs = 0
                best_state = {key: value.detach().cpu().clone() for key, value in module.state_dict().items()}
            else:
                stale_epochs += 1
            if (epoch + 1) >= min_epochs and stale_epochs >= patience:
                break
        module.load_state_dict(best_state)
        module = module.to("cpu")
        return (
            {"kind": "torch", "module": module},
            {
                "device": str(device),
                "epochs_trained": epoch + 1,
                "best_epoch": best_epoch,
                "best_stop_macro_f1": round(best_score, 4),
                "stopped_early": stale_epochs >= patience,
            },
        )

    def _predict(self, model: dict[str, object], features: list[float]) -> int:
        if model["kind"] == "linear":
            logits = [self._dot(weight, features) for weight in model["weights"]]
        elif model["kind"] == "torch":
            assert self._torch is not None
            with self._torch.no_grad():
                tensor = self._torch.tensor(features, dtype=self._torch.float32).unsqueeze(0)
                logits = model["module"](tensor).squeeze(0).tolist()
        else:
            hidden = [math.tanh(self._dot(weight, features)) for weight in model["hidden"]]
            logits = [self._dot(weight, hidden) for weight in model["output"]]
        return max(range(len(logits)), key=lambda index: logits[index])

    def _parameter_count(self, model: dict[str, object]) -> int:
        if model["kind"] == "linear":
            return len(model["weights"]) * len(model["weights"][0])
        if model["kind"] == "torch":
            return sum(parameter.numel() for parameter in model["module"].parameters())
        hidden = model["hidden"]
        output = model["output"]
        return (len(hidden) * len(hidden[0])) + (len(output) * len(output[0]))

    def _resolve_device(self, device_preference: str) -> str:
        if self._torch is None:
            return "cpu"
        if device_preference == "cuda" and self._torch.cuda.is_available():
            return "cuda"
        if device_preference == "cpu":
            return "cpu"
        return "cuda" if self._torch.cuda.is_available() else "cpu"

    def _split_inner_validation(
        self,
        features: list[list[float]],
        labels: list[int],
    ) -> tuple[list[list[float]], list[int], list[list[float]], list[int]]:
        grouped: dict[int, list[tuple[list[float]], int]] = {}
        for feature, label in zip(features, labels):
            grouped.setdefault(label, []).append((feature, label))
        train_features: list[list[float]] = []
        train_labels: list[int] = []
        stop_features: list[list[float]] = []
        stop_labels: list[int] = []
        for items in grouped.values():
            split_index = max(1, len(items) // 5)
            stop_items = items[:split_index]
            train_items = items[split_index:] or items[:1]
            for feature, label in train_items:
                train_features.append(feature)
                train_labels.append(label)
            for feature, label in stop_items:
                stop_features.append(feature)
                stop_labels.append(label)
        return train_features, train_labels, stop_features, stop_labels

    def _multiscale_features(self, window: list[list[float]]) -> list[float]:
        channels = len(window[0])
        features: list[float] = []
        segment_counts = [2, 4]
        for segment_count in segment_counts:
            segment_size = max(len(window) // segment_count, 1)
            for start in range(0, len(window), segment_size):
                segment = window[start : start + segment_size]
                if not segment:
                    continue
                features.extend(
                    sum(row[channel] for row in segment) / len(segment)
                    for channel in range(channels)
                )
        return features

    def _saliency_features(self, window: list[list[float]]) -> list[float]:
        channels = len(window[0])
        motion = [
            sum(abs(window[index][channel] - window[index - 1][channel]) for channel in range(channels))
            for index in range(1, len(window))
        ]
        if not motion:
            return [0.0 for _ in range(channels * 2)]
        top_indices = sorted(range(len(motion)), key=lambda index: motion[index], reverse=True)[: max(2, len(motion) // 8)]
        salient_rows = [window[index + 1] for index in top_indices]
        pooled = [
            sum(row[channel] for row in salient_rows) / len(salient_rows)
            for channel in range(channels)
        ]
        peak = [max(abs(row[channel]) for row in salient_rows) for channel in range(channels)]
        return pooled + peak

    def _interaction_features(self, window: list[list[float]]) -> list[float]:
        channels = len(window[0])
        interactions: list[float] = []
        for left in range(min(channels, 4)):
            for right in range(left + 1, min(channels, 4)):
                interactions.append(
                    sum(row[left] * row[right] for row in window) / len(window)
                )
        return interactions

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

    def _checkpoint_payload(
        self,
        model: dict[str, object],
        config: CandidateConfig,
        training_report: dict[str, object],
    ) -> dict[str, object]:
        payload = {
            "kind": str(model["kind"]),
            "config": {
                "model_name": config.model_name,
                "preprocessing": config.preprocessing,
                "augmentation": config.augmentation,
                "augmentation_strength": config.augmentation_strength,
                "learning_rate": config.learning_rate,
                "window_size": config.window_size,
            },
            "training_report": dict(training_report),
        }
        if model["kind"] == "linear":
            payload["weights"] = model["weights"]
        elif model["kind"] == "mlp":
            payload["hidden"] = model["hidden"]
            payload["output"] = model["output"]
        elif model["kind"] == "torch":
            assert self._torch is not None
            payload["state_dict"] = {
                key: value.detach().cpu()
                for key, value in model["module"].state_dict().items()
            }
        return payload

    def _write_checkpoint_artifact(
        self,
        round_id: int,
        agent_name: str,
        branch_name: str,
        checkpoint_payload: dict[str, object],
    ) -> str:
        round_dir = ensure_round_dir(round_id)
        checkpoint_dir = round_dir / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        if checkpoint_payload["kind"] == "torch":
            checkpoint_path = checkpoint_dir / f"{agent_name}.pt"
            assert self._torch is not None
            self._torch.save(
                {
                    "agent_name": agent_name,
                    "branch_name": branch_name,
                    **checkpoint_payload,
                },
                checkpoint_path,
            )
            metadata_path = checkpoint_dir / f"{agent_name}.json"
            metadata_payload = {
                "agent_name": agent_name,
                "branch_name": branch_name,
                "kind": checkpoint_payload["kind"],
                "config": checkpoint_payload["config"],
                "training_report": checkpoint_payload["training_report"],
                "checkpoint_file": str(checkpoint_path),
            }
            write_json_artifact(metadata_path, metadata_payload)
            return str(checkpoint_path)
        checkpoint_path = checkpoint_dir / f"{agent_name}.json"
        checkpoint_payload = {
            "agent_name": agent_name,
            "branch_name": branch_name,
            **checkpoint_payload,
        }
        return write_json_artifact(checkpoint_path, checkpoint_payload)
