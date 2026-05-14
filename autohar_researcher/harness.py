from __future__ import annotations

import importlib
import json
import math
import random
import re
from pathlib import Path

try:
    import torch
except Exception:  # pragma: no cover - optional dependency path
    torch = None

from autohar_researcher.artifacts import ensure_round_dir, write_json_artifact
from autohar_researcher.datasets import WindowedHARDataset, create_har_dataset
from autohar_researcher.sandbox import sandboxed_namespace
from autohar_researcher.types import CandidateConfig, CandidateResult


class DummyHARHarness:
    """Small but real HAR train/validation pipeline using UCI HAR."""

    LABELS = ["walk", "walk_upstairs", "walk_downstairs", "sit", "stand", "lay"]

    def __init__(self) -> None:
        self._dataset: WindowedHARDataset | None = None
        self._dataset_name: str | None = None
        self._dataset_cache: dict[str, WindowedHARDataset] = {}
        self._evaluation_cache: dict[tuple[object, ...], dict[str, object]] = {}
        self._custom_augmentation_cache: dict[str, object | None] = {}
        self._custom_model_feature_cache: dict[str, object | None] = {}
        self._custom_data_cache: dict[str, object | None] = {}
        self._torch = torch

    def runtime_backend_summary(self) -> str:
        if self._torch is None:
            return "PyTorch not available; using CPU fallback path."
        if self._torch.cuda.is_available():
            return "PyTorch available; CUDA GPU detected and will be used when training device is auto."
        return "PyTorch available; no GPU found, using CPU."

    def _copy_window(self, window: list[list[float]]) -> list[list[float]]:
        return [row[:] for row in window]

    def _safe_custom_params(
        self,
        params: dict[str, float | int | str | bool],
    ) -> dict[str, float | int | str | bool]:
        return {key: value for key, value in params.items() if key != "custom_code"}

    def _cached_custom_callable(
        self,
        cache: dict[str, object | None],
        custom_code: str,
        function_name: str,
    ) -> object | None:
        if custom_code not in cache:
            namespace = sandboxed_namespace(custom_code) or {}
            candidate = namespace.get(function_name)
            cache[custom_code] = candidate if callable(candidate) else None
        return cache[custom_code]

    def _candidate_summary(
        self,
        agent_name: str,
        config: CandidateConfig,
    ) -> str:
        return (
            f"{agent_name} proposed {config.model_name} with {config.preprocessing}, "
            f"{config.augmentation}:{config.augmentation_strength}, "
            f"lr={config.learning_rate}, window={config.window_size}, "
            f"model_params={config.model_params}, data_params={config.data_params}, "
            f"aug_params={config.augmentation_params}"
        )

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
        evaluation_tag: str = "",
    ) -> CandidateResult:
        experiment = experiment or self._load_experiment(Path(worktree_path))
        train_dataset_names, eval_dataset_name, synthetic_sources = self._resolve_dataset_plan(
            experiment,
            config.data_params,
        )
        cache_key = (
            tuple(train_dataset_names),
            eval_dataset_name,
            tuple(synthetic_sources),
            config.model_name,
            json.dumps(config.model_params, sort_keys=True),
            config.preprocessing,
            json.dumps(config.data_params, sort_keys=True),
            config.augmentation,
            config.augmentation_strength,
            json.dumps(config.augmentation_params, sort_keys=True),
            float(config.learning_rate),
            int(config.window_size),
            str(evaluation_tag),
        )
        if cache_key not in self._evaluation_cache:
            eval_dataset = self._load_dataset_by_name(eval_dataset_name)
            training = self._training_config(experiment)
            raw_train_samples = self._load_training_samples(
                train_dataset_names,
                config.data_params,
            )
            raw_val_samples = eval_dataset.load_split("test")
            train_samples, val_samples, common_labels = self._align_dataset_classes(
                train_samples=raw_train_samples,
                val_samples=raw_val_samples,
                data_params=config.data_params,
            )
            if not train_samples or not val_samples:
                raise RuntimeError(
                    "No overlapping classes remained after dataset/class filtering "
                    f"for train={train_dataset_names!r}, eval={eval_dataset_name!r}."
                )
            self.LABELS = list(common_labels)
            label_to_index = {label: index for index, label in enumerate(self.LABELS)}
            train_samples = [
                (self._prepare_window(window, config, split="train"), label)
                for window, label in train_samples
            ]
            val_samples = [
                (self._prepare_window(window, config, split="test"), label)
                for window, label in val_samples
            ]
            train_samples = self._add_synthetic_training_samples(
                train_samples,
                synthetic_sources,
                config,
            )

            train_features = [
                self._extract_features(window, config) for window, _ in train_samples
            ]
            train_labels = [label_to_index[label] for _, label in train_samples]
            val_features = [
                self._extract_features(window, config) for window, _ in val_samples
            ]
            val_labels = [label_to_index[label] for _, label in val_samples]
            inner_train_features, inner_train_labels, stop_features, stop_labels = (
                self._split_inner_validation(
                    train_features,
                    train_labels,
                )
            )

            model, training_report = self._train_model(
                model_name=config.model_name,
                model_params=config.model_params,
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
                    "dataset_alignment": {
                        "train_datasets": list(train_dataset_names),
                        "eval_dataset": eval_dataset_name,
                        "common_classes": list(common_labels),
                        "class_policy": str(
                            config.data_params.get("class_sampling_policy", "preserve")
                        ),
                        "target_classes": config.data_params.get(
                            "target_classes", "all"
                        ),
                        "synthetic_sources": list(synthetic_sources),
                        "synthetic_target_classes": config.data_params.get(
                            "synthetic_target_classes", "all"
                        ),
                        "synthetic_source_targets": config.data_params.get(
                            "synthetic_source_targets", ""
                        ),
                    },
                    "confusion_matrix": confusion_matrix,
                    "per_class_f1": per_class_f1,
                    "training_report": training_report,
                },
            }
        cached = self._evaluation_cache[cache_key]
        summary = self._candidate_summary(agent_name=agent_name, config=config)
        details = dict(cached["details"])
        score_breakdown = {
            key: value for key, value in cached.items() if isinstance(value, float)
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

    def _resolve_dataset_plan(
        self,
        experiment: dict[str, object],
        data_params: dict[str, float | int | str | bool],
    ) -> tuple[list[str], str, list[str]]:
        experiment_dataset = str(experiment.get("dataset", {}).get("name", "uci_har"))
        policy = str(data_params.get("dataset_policy", "single_dataset")).strip().lower()
        source_names = self._split_config_list(data_params.get("dataset_sources", ""))
        train_dataset_names = self._split_config_list(data_params.get("train_datasets", ""))
        synthetic_sources = [
            source for source in self._split_config_list(data_params.get("synthetic_sources", ""))
            if source.startswith("synthetic_")
        ]
        synthetic_sources.extend(
            source for source in source_names if source.startswith("synthetic_")
        )
        synthetic_sources = list(dict.fromkeys(synthetic_sources))
        real_sources = [source for source in source_names if not source.startswith("synthetic_")]
        train_datasets = train_dataset_names or [experiment_dataset]
        if policy in {"multi_dataset", "cross_dataset", "cross_dataset_eval", "real_synthetic_mix"}:
            train_datasets.extend(real_sources)
        train_datasets = list(dict.fromkeys(name for name in train_datasets if name))
        eval_dataset = str(
            data_params.get("evaluation_dataset")
            or data_params.get("eval_dataset")
            or data_params.get("test_dataset")
            or ""
        ).strip()
        if not eval_dataset:
            eval_dataset = train_datasets[0] if train_datasets else experiment_dataset
        return train_datasets, eval_dataset, synthetic_sources

    def _load_training_samples(
        self,
        dataset_names: list[str],
        data_params: dict[str, float | int | str | bool],
    ) -> list[tuple[list[list[float]], str]]:
        samples: list[tuple[list[list[float]], str]] = []
        per_dataset_ratio = self._ratio_mapping(data_params.get("per_dataset_real_ratio", ""))
        for dataset_name in dataset_names:
            dataset = self._load_dataset_by_name(dataset_name)
            dataset_samples = dataset.load_split("train")
            dataset_samples = self._limit_samples_by_ratio(
                dataset_samples,
                per_dataset_ratio.get(dataset_name, data_params.get("real_ratio", 1.0)),
            )
            dataset_samples = self._limit_samples_by_class_ratio(
                dataset_samples,
                data_params.get("per_class_real_ratio", ""),
            )
            dataset_samples = self._limit_samples_by_dataset_class_ratio(
                dataset_samples,
                dataset_name,
                data_params.get("per_dataset_class_real_ratio", ""),
            )
            samples.extend(dataset_samples)
        return samples

    def _limit_samples_by_ratio(
        self,
        samples: list[tuple[list[list[float]], str]],
        ratio_value: object,
    ) -> list[tuple[list[list[float]], str]]:
        try:
            ratio = float(ratio_value)
        except (TypeError, ValueError):
            ratio = 1.0
        if ratio >= 1.0:
            return samples
        if ratio <= 0.0:
            return []
        grouped: dict[str, list[tuple[list[list[float]], str]]] = {}
        for sample in samples:
            grouped.setdefault(sample[1], []).append(sample)
        limited: list[tuple[list[list[float]], str]] = []
        for label in sorted(grouped):
            group = grouped[label]
            keep = max(1, int(round(len(group) * ratio)))
            limited.extend(group[:keep])
        return limited

    def _limit_samples_by_class_ratio(
        self,
        samples: list[tuple[list[list[float]], str]],
        ratio_value: object,
    ) -> list[tuple[list[list[float]], str]]:
        ratios = self._ratio_mapping(ratio_value)
        if not ratios:
            return samples
        grouped: dict[str, list[tuple[list[list[float]], str]]] = {}
        for sample in samples:
            grouped.setdefault(self._canonical_activity_label(sample[1]), []).append(sample)
        limited: list[tuple[list[list[float]], str]] = []
        for label, group in grouped.items():
            ratio = ratios.get(label, 1.0)
            keep = len(group) if ratio >= 1.0 else max(1, int(round(len(group) * max(0.0, ratio))))
            limited.extend(group[:keep])
        return limited

    def _limit_samples_by_dataset_class_ratio(
        self,
        samples: list[tuple[list[list[float]], str]],
        dataset_name: str,
        ratio_value: object,
    ) -> list[tuple[list[list[float]], str]]:
        mapping = self._parse_mapping(ratio_value)
        raw_dataset_ratios = mapping.get(dataset_name)
        if raw_dataset_ratios is None:
            return samples
        return self._limit_samples_by_class_ratio(samples, raw_dataset_ratios)

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
                "dataset": {"name": "uci_har"},
                "training": {"epochs": 5},
            }
        payload = json.loads(experiment_path.read_text(encoding="utf-8"))
        if "experiment" in payload:
            return payload["experiment"]
        return payload

    def _load_dataset(self, experiment: dict[str, object]) -> WindowedHARDataset:
        dataset_name = str(experiment["dataset"]["name"])
        return self._load_dataset_by_name(dataset_name)

    def _load_dataset_by_name(self, dataset_name: str) -> WindowedHARDataset:
        if self._dataset is None or self._dataset_name != dataset_name:
            if dataset_name not in self._dataset_cache:
                dataset = create_har_dataset(dataset_name)
                dataset.ensure_downloaded()
                self._dataset_cache[dataset_name] = dataset
            self._dataset = self._dataset_cache[dataset_name]
            self._dataset_name = dataset_name
        return self._dataset

    def _align_dataset_classes(
        self,
        train_samples: list[tuple[list[list[float]], str]],
        val_samples: list[tuple[list[list[float]], str]],
        data_params: dict[str, float | int | str | bool],
    ) -> tuple[
        list[tuple[list[list[float]], str]],
        list[tuple[list[list[float]], str]],
        list[str],
    ]:
        train_by_canonical = self._label_display_by_canonical(train_samples)
        val_by_canonical = self._label_display_by_canonical(val_samples)
        eval_targets = set(val_by_canonical)
        targets = self._target_class_set(data_params)
        if targets is not None:
            eval_targets &= targets
        common = set(train_by_canonical).intersection(eval_targets)
        ordered_common = [
            canonical
            for canonical in val_by_canonical
            if canonical in common
        ]
        canonical_to_label = {
            canonical: val_by_canonical[canonical]
            for canonical in ordered_common
        }
        aligned_train = self._filter_and_relabel_samples(
            train_samples,
            canonical_to_label,
        )
        aligned_val = self._filter_and_relabel_samples(
            val_samples,
            canonical_to_label,
        )
        labels = [canonical_to_label[canonical] for canonical in ordered_common]
        return aligned_train, aligned_val, labels

    def _label_display_by_canonical(
        self,
        samples: list[tuple[list[list[float]], str]],
    ) -> dict[str, str]:
        labels: dict[str, str] = {}
        for _, label in samples:
            canonical = self._canonical_activity_label(label)
            labels.setdefault(canonical, canonical)
        return labels

    def _filter_and_relabel_samples(
        self,
        samples: list[tuple[list[list[float]], str]],
        canonical_to_label: dict[str, str],
    ) -> list[tuple[list[list[float]], str]]:
        filtered: list[tuple[list[list[float]], str]] = []
        for window, label in samples:
            canonical = self._canonical_activity_label(label)
            if canonical in canonical_to_label:
                filtered.append((window, canonical_to_label[canonical]))
        return filtered

    def _target_class_set(
        self,
        data_params: dict[str, float | int | str | bool],
    ) -> set[str] | None:
        raw_targets = data_params.get("target_classes", "all")
        if raw_targets in {"", "all", None}:
            return None
        targets = {
            self._canonical_activity_label(target)
            for target in self._split_config_list(raw_targets)
        }
        return targets or None

    def _split_config_list(self, value: object) -> list[str]:
        if value is None:
            return []
        if isinstance(value, (list, tuple, set)):
            return [str(item).strip() for item in value if str(item).strip()]
        text = str(value).strip()
        if not text:
            return []
        if text.startswith("[") and text.endswith("]"):
            try:
                parsed = json.loads(text.replace("'", '"'))
                if isinstance(parsed, list):
                    return [str(item).strip() for item in parsed if str(item).strip()]
            except json.JSONDecodeError:
                pass
        return [part.strip() for part in text.split(",") if part.strip()]

    def _canonical_activity_label(self, label: object) -> str:
        token = re.sub(r"[^a-z0-9]+", "_", str(label).strip().lower()).strip("_")
        aliases = {
            "walking": "walk",
            "walk": "walk",
            "jogging": "run",
            "running": "run",
            "run": "run",
            "sitting": "sit",
            "sit": "sit",
            "standing": "stand",
            "stand": "stand",
            "lying": "lie",
            "laying": "lie",
            "lay": "lie",
            "lie": "lie",
            "upstairs": "walk_upstairs",
            "walking_upstairs": "walk_upstairs",
            "walk_upstairs": "walk_upstairs",
            "ascending_stairs": "walk_upstairs",
            "climbing_stairs": "stairs",
            "stairs_up": "walk_upstairs",
            "downstairs": "walk_downstairs",
            "walking_downstairs": "walk_downstairs",
            "walk_downstairs": "walk_downstairs",
            "descending_stairs": "walk_downstairs",
            "stairs_down": "walk_downstairs",
        }
        return aliases.get(token, token)

    def _add_synthetic_training_samples(
        self,
        train_samples: list[tuple[list[list[float]], str]],
        synthetic_sources: list[str],
        config: CandidateConfig,
    ) -> list[tuple[list[list[float]], str]]:
        if not train_samples or not synthetic_sources:
            return train_samples
        try:
            synthetic_ratio = float(config.data_params.get("synthetic_ratio", 0.0))
        except (TypeError, ValueError):
            synthetic_ratio = 0.0
        if synthetic_ratio <= 0.0:
            return train_samples

        source_modes = {
            "synthetic_imu_to_imu": "imu_to_imu",
            "synthetic_text_to_imu": "text_to_imu",
            "synthetic_video_to_imu": "video_to_imu",
        }
        augmented = list(train_samples)
        source_targets = self._synthetic_source_targets(config.data_params)
        source_ratios = self._per_source_synthetic_ratios(config.data_params)
        global_targets = self._target_class_set(
            {"target_classes": config.data_params.get("synthetic_target_classes", "all")}
        )

        for source_index, source in enumerate(synthetic_sources):
            mode = source_modes.get(source, source)
            source_target_set = source_targets.get(source) or source_targets.get(mode)
            if source_target_set is None:
                source_target_set = global_targets
            source_samples = self._samples_for_target_classes(
                train_samples,
                source_target_set,
            )
            if not source_samples:
                continue
            source_ratio = source_ratios.get(source, source_ratios.get(mode, synthetic_ratio))
            target_count = int(round(len(source_samples) * source_ratio))
            if target_count <= 0:
                continue
            for index in range(target_count):
                window, label = source_samples[index % len(source_samples)]
                params = dict(config.augmentation_params)
                params.setdefault("synthetic_source", mode)
                params.setdefault("synthetic_target_class", label)
                rng = random.Random(config.window_size * 1009 + source_index * 104729 + index)
                synthetic_window = self._augment(
                    self._copy_window(window),
                    mode,
                    config.augmentation_strength,
                    params,
                    rng=rng,
                )
                augmented.append((synthetic_window, label))
        return augmented

    def _samples_for_target_classes(
        self,
        samples: list[tuple[list[list[float]], str]],
        target_classes: set[str] | None,
    ) -> list[tuple[list[list[float]], str]]:
        if target_classes is None:
            return samples
        return [
            sample
            for sample in samples
            if self._canonical_activity_label(sample[1]) in target_classes
        ]

    def _synthetic_source_targets(
        self,
        data_params: dict[str, float | int | str | bool],
    ) -> dict[str, set[str]]:
        raw = data_params.get("synthetic_source_targets", "")
        parsed = self._parse_mapping(raw)
        targets: dict[str, set[str]] = {}
        for source, value in parsed.items():
            class_set = {
                self._canonical_activity_label(item)
                for item in self._split_config_list(value)
            }
            if class_set:
                targets[source] = class_set
        return targets

    def _per_source_synthetic_ratios(
        self,
        data_params: dict[str, float | int | str | bool],
    ) -> dict[str, float]:
        parsed = self._parse_mapping(data_params.get("per_source_synthetic_ratio", ""))
        return self._coerce_ratio_mapping(parsed)

    def _ratio_mapping(self, value: object) -> dict[str, float]:
        parsed = self._parse_mapping(value)
        return self._coerce_ratio_mapping(parsed)

    def _coerce_ratio_mapping(self, parsed: dict[str, object]) -> dict[str, float]:
        ratios: dict[str, float] = {}
        for source, value in parsed.items():
            try:
                ratios[self._canonical_activity_label(source)] = max(0.0, float(value))
            except (TypeError, ValueError):
                continue
        return ratios

    def _parse_mapping(self, value: object) -> dict[str, object]:
        if value is None or value == "":
            return {}
        if isinstance(value, dict):
            return {str(key): item for key, item in value.items()}
        text = str(value).strip()
        if not text:
            return {}
        try:
            parsed = json.loads(text.replace("'", '"'))
            if isinstance(parsed, dict):
                return {str(key): item for key, item in parsed.items()}
        except json.JSONDecodeError:
            pass
        result: dict[str, object] = {}
        for part in text.split(";"):
            if ":" not in part:
                continue
            key, raw_value = part.split(":", 1)
            if key.strip():
                result[key.strip()] = raw_value.strip()
        return result

    def _prepare_window(
        self,
        window: list[list[float]],
        config: CandidateConfig,
        split: str,
    ) -> list[list[float]]:
        processed = self._select_channels(
            window, config.preprocessing, config.data_params
        )
        processed = self._resample_window(
            processed, config.window_size, config.data_params
        )
        processed = self._preprocess(
            processed, config.preprocessing, config.data_params
        )
        if split == "train":
            rng = random.Random(config.window_size + len(window))
            processed = self._augment(
                processed,
                config.augmentation,
                config.augmentation_strength,
                config.augmentation_params,
                rng=rng,
            )
        return processed

    def _select_channels(
        self,
        window: list[list[float]],
        mode: str,
        params: dict[str, float | int | str | bool],
    ) -> list[list[float]]:
        selected: list[list[float]] = []
        for row in window:
            total_acc = row[:3]
            body_acc = row[3:6]
            body_gyro = row[6:9]
            channel_layout = str(
                params.get(
                    "channel_layout",
                    "body_acc+gyro"
                    if mode == "normalized+gravity_filter"
                    else "total_acc+gyro",
                )
            )
            if channel_layout == "body_acc+gyro":
                selected.append(body_acc + body_gyro)
            else:
                selected.append(total_acc + body_gyro)
        return selected

    def _resample_window(
        self,
        window: list[list[float]],
        target_size: int,
        params: dict[str, float | int | str | bool],
    ) -> list[list[float]]:
        if len(window) == target_size:
            return self._copy_window(window)
        resample_mode = str(params.get("resample_mode", "nearest"))
        resampled = []
        for idx in range(target_size):
            position = idx * (len(window) - 1) / max(target_size - 1, 1)
            if resample_mode == "linear":
                left = int(math.floor(position))
                right = min(left + 1, len(window) - 1)
                mix = position - left
                resampled.append(
                    [
                        ((1.0 - mix) * window[left][channel])
                        + (mix * window[right][channel])
                        for channel in range(len(window[0]))
                    ]
                )
            else:
                source = int(position)
                resampled.append(window[source][:])
        return resampled

    def _preprocess(
        self,
        window: list[list[float]],
        mode: str,
        params: dict[str, float | int | str | bool],
    ) -> list[list[float]]:
        processed = [row[:] for row in window]
        if mode == "custom":
            return self._custom_prepare_window(processed, params)
        module_processed = self._preprocess_from_module(processed, mode, params)
        if module_processed is not None:
            return module_processed
        if bool(
            params.get(
                "normalize",
                mode
                in {
                    "normalized",
                    "normalized+gravity_filter",
                    "normalized+magnitude_channels",
                },
            )
        ):
            processed = self._normalize(processed)
        if mode == "normalized+gravity_filter":
            gravity_scale = float(params.get("gravity_scale", 1.0))
            accel_means = [
                sum(row[channel] for row in processed) / len(processed)
                for channel in range(3)
            ]
            for row in processed:
                for channel in range(3):
                    row[channel] -= gravity_scale * accel_means[channel]
        if bool(
            params.get("append_magnitude", mode == "normalized+magnitude_channels")
        ):
            extended: list[list[float]] = []
            for row in processed:
                accel_mag = math.sqrt(sum(value * value for value in row[:3]))
                gyro_mag = math.sqrt(sum(value * value for value in row[3:6]))
                extended.append(row + [accel_mag, gyro_mag])
            processed = extended
        return processed

    def _preprocess_from_module(
        self,
        window: list[list[float]],
        mode: str,
        params: dict[str, float | int | str | bool],
    ) -> list[list[float]] | None:
        module_name = mode.replace("+", "_").replace("-", "_")
        try:
            module = importlib.import_module(
                f"research.modules.preprocessing.{module_name}"
            )
        except Exception:
            return None
        prepare_fn = getattr(module, "prepare_window", None)
        if not callable(prepare_fn):
            return None
        try:
            candidate = prepare_fn(self._copy_window(window), dict(params))
        except Exception:
            return None
        return self._validated_custom_window(candidate, window)

    def _custom_prepare_window(
        self,
        window: list[list[float]],
        params: dict[str, float | int | str | bool],
    ) -> list[list[float]]:
        custom_code = str(params.get("custom_code", "")).strip()
        if not custom_code:
            return self._copy_window(window)
        prepare_fn = self._custom_data_callable(custom_code)
        if not callable(prepare_fn):
            return self._copy_window(window)
        try:
            result = prepare_fn(
                self._copy_window(window),
                dict(self._safe_custom_params(params)),
            )
        except Exception:  # noqa: BLE001
            return self._copy_window(window)
        return self._validated_custom_window(result, window)

    def _custom_data_callable(self, custom_code: str) -> object | None:
        return self._cached_custom_callable(
            cache=self._custom_data_cache,
            custom_code=custom_code,
            function_name="custom_prepare_window",
        )

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
        params: dict[str, float | int | str],
        rng: random.Random,
    ) -> list[list[float]]:
        augmented = self._copy_window(window)
        strength_scale = {
            "weak": 0.5,
            "medium": 1.0,
            "strong": 1.5,
        }.get(strength, 1.0)
        if mode == "none":
            return augmented
        if mode == "custom":
            return self._custom_augment(
                augmented,
                strength=strength,
                params=params,
                rng=rng,
            )
        module_augmented = self._augment_from_module(
            augmented,
            mode=mode,
            strength=strength,
            params=params,
            rng=rng,
        )
        if module_augmented is not None:
            return module_augmented
        if mode == "jitter":
            radius = float(params.get("jitter_std", 0.02 * strength_scale))
            return [
                [value + rng.uniform(-radius, radius) for value in row]
                for row in augmented
            ]
        if mode == "scaling":
            span = float(params.get("scale_span", 0.1 * strength_scale))
            scales = [
                rng.uniform(1.0 - span, 1.0 + span) for _ in range(len(augmented[0]))
            ]
            return [
                [value * scales[index] for index, value in enumerate(row)]
                for row in augmented
            ]
        if mode == "rotation":
            angle_deg = float(params.get("angle_deg", 10.0 * strength_scale))
            angle = math.radians(angle_deg)
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
            compression = float(
                params.get("compression_ratio", max(0.7, 1.0 - (0.08 * strength_scale)))
            )
            for index in range(len(augmented)):
                source_index = min(len(augmented) - 1, int(index * compression))
                warped.append(augmented[source_index][:])
            return warped
        if mode == "permutation":
            if len(augmented) < 8:
                return augmented
            segment_count = int(
                params.get(
                    "segment_count",
                    min(6, max(3, int(round(2 + (2 * strength_scale))))),
                )
            )
            segment_size = max(len(augmented) // segment_count, 1)
            segments = [
                augmented[index : index + segment_size]
                for index in range(0, len(augmented), segment_size)
            ]
            rng.shuffle(segments)
            return [row[:] for segment in segments for row in segment]
        if mode == "drift":
            drift_radius = float(params.get("drift_max", 0.015 * strength_scale))
            drifts = [
                rng.uniform(-drift_radius, drift_radius)
                for _ in range(len(augmented[0]))
            ]
            drifted: list[list[float]] = []
            for step, row in enumerate(augmented):
                alpha = step / max(len(augmented) - 1, 1)
                drifted.append(
                    [value + (alpha * drifts[index]) for index, value in enumerate(row)]
                )
            return drifted
        if mode == "magnitude_warp":
            anchors = [0.0, 0.33, 0.66, 1.0]
            span = float(params.get("warp_span", 0.15 * strength_scale))
            knots = [rng.uniform(1.0 - span, 1.0 + span) for _ in anchors]
            warped: list[list[float]] = []
            for step, row in enumerate(augmented):
                alpha = step / max(len(augmented) - 1, 1)
                left = max(
                    index for index, anchor in enumerate(anchors) if anchor <= alpha
                )
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
            blend = float(
                params.get("blend", min(0.45, 0.15 + (0.15 * strength_scale)))
            )
            mixed = []
            reversed_window = list(reversed(augmented))
            for left, right in zip(augmented, reversed_window):
                mixed.append(
                    [((1.0 - blend) * a) + (blend * b) for a, b in zip(left, right)]
                )
            return mixed
        if mode == "channel_dropout":
            probability = float(params.get("drop_probability", 0.08 * strength_scale))
            channel_count = len(augmented[0]) if augmented else 0
            dropped = {
                channel
                for channel in range(channel_count)
                if rng.uniform(0.0, 1.0) < probability
            }
            if not dropped and channel_count:
                dropped.add(int(rng.uniform(0, channel_count)))
            return [
                [
                    0.0 if channel in dropped else value
                    for channel, value in enumerate(row)
                ]
                for row in augmented
            ]
        if mode == "sensor_bias":
            radius = float(params.get("bias_radius", 0.02 * strength_scale))
            biases = [rng.uniform(-radius, radius) for _ in range(len(augmented[0]))]
            return [
                [value + biases[channel] for channel, value in enumerate(row)]
                for row in augmented
            ]
        if mode == "frequency_mask":
            ratio = max(0.0, min(0.6, float(params.get("mask_ratio", 0.12 * strength_scale))))
            if len(augmented) < 4:
                return augmented
            span = max(1, int(len(augmented) * ratio))
            start = int(rng.uniform(0, max(1, len(augmented) - span)))
            masked = self._copy_window(augmented)
            channel_means = [
                sum(row[channel] for row in augmented) / len(augmented)
                for channel in range(len(augmented[0]))
            ]
            for idx in range(start, min(len(masked), start + span)):
                masked[idx] = channel_means[:]
            return masked
        if mode in {"imu_to_imu", "text_to_imu", "video_to_imu"}:
            return self._placeholder_generative_augment(
                augmented,
                mode=mode,
                strength_scale=strength_scale,
                params=params,
                rng=rng,
            )
        return augmented

    def _augment_from_module(
        self,
        window: list[list[float]],
        mode: str,
        strength: str,
        params: dict[str, float | int | str],
        rng: random.Random,
    ) -> list[list[float]] | None:
        safe_name = mode.replace("-", "_")
        try:
            module = importlib.import_module(
                f"research.modules.augmentations.{safe_name}"
            )
        except Exception:
            return None
        augment_fn = getattr(module, "augment_window", None)
        if not callable(augment_fn):
            return None
        try:
            candidate = augment_fn(self._copy_window(window), strength, dict(params), rng)
        except Exception:
            return None
        return self._validated_custom_window(candidate, window)

    def _placeholder_generative_augment(
        self,
        window: list[list[float]],
        mode: str,
        strength_scale: float,
        params: dict[str, float | int | str],
        rng: random.Random,
    ) -> list[list[float]]:
        del mode
        amount = max(
            0.0,
            min(0.5, float(params.get("variation_amount", 0.10 * strength_scale))),
        )
        noise_radius = amount * 0.05
        bias_radius = amount * 0.03
        biases = [rng.uniform(-bias_radius, bias_radius) for _ in range(len(window[0]))]
        generated: list[list[float]] = []
        for step, row in enumerate(window):
            alpha = step / max(len(window) - 1, 1)
            slow_wave = math.sin(alpha * math.pi * 2.0) * amount * 0.02
            generated.append(
                [
                    value
                    + biases[channel]
                    + slow_wave
                    + rng.uniform(-noise_radius, noise_radius)
                    for channel, value in enumerate(row)
                ]
            )
        return generated

    def _custom_augment(
        self,
        window: list[list[float]],
        strength: str,
        params: dict[str, float | int | str],
        rng: random.Random,
    ) -> list[list[float]]:
        custom_code = str(params.get("custom_code", "")).strip()
        if not custom_code:
            return self._copy_window(window)
        augment_fn = self._custom_augmentation_callable(custom_code)
        if not callable(augment_fn):
            return self._copy_window(window)
        try:
            result = augment_fn(
                self._copy_window(window),
                str(strength),
                dict(self._safe_custom_params(params)),
                rng,
            )
        except Exception:  # noqa: BLE001
            return self._copy_window(window)
        return self._validated_custom_window(result, window)

    def _custom_augmentation_callable(self, custom_code: str) -> object | None:
        return self._cached_custom_callable(
            cache=self._custom_augmentation_cache,
            custom_code=custom_code,
            function_name="custom_augment_window",
        )

    def _validated_custom_window(
        self,
        candidate: object,
        fallback: list[list[float]],
    ) -> list[list[float]]:
        if not isinstance(candidate, list) or len(candidate) != len(fallback):
            return self._copy_window(fallback)
        channel_count = len(fallback[0]) if fallback else 0
        validated: list[list[float]] = []
        for row in candidate:
            if not isinstance(row, list) or len(row) != channel_count:
                return self._copy_window(fallback)
            clean_row: list[float] = []
            for value in row:
                if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                    return self._copy_window(fallback)
                clean_row.append(float(value))
            validated.append(clean_row)
        return validated

    def _extract_features(
        self, window: list[list[float]], config: CandidateConfig
    ) -> list[float]:
        channels = len(window[0])
        model_params = config.model_params
        temporal_pool = str(model_params.get("temporal_pool", "avg"))
        spectral_bins = max(2, int(model_params.get("spectral_bins", 6)))
        use_interactions = bool(model_params.get("use_interactions", True))
        use_multiscale = bool(model_params.get("use_multiscale", True))
        attention_mix = max(
            0.0, min(1.0, float(model_params.get("attention_mix", 0.5)))
        )
        means = [sum(row[i] for row in window) / len(window) for i in range(channels)]
        stds = []
        energies = []
        diffs = []
        for i in range(channels):
            variance = sum((row[i] - means[i]) ** 2 for row in window) / len(window)
            stds.append(math.sqrt(variance))
            energies.append(sum(row[i] * row[i] for row in window) / len(window))
            diffs.append(
                sum(
                    abs(window[idx][i] - window[idx - 1][i])
                    for idx in range(1, len(window))
                )
                / max(len(window) - 1, 1)
            )

        multiscale = self._multiscale_features(window) if use_multiscale else []
        saliency = self._saliency_features(window)
        interactions = self._interaction_features(window) if use_interactions else []
        temporal = self._temporal_pool_features(window, temporal_pool)
        spectral = self._spectral_features(window, spectral_bins)
        attention = self._attention_pool_features(window, attention_mix)

        default_features = means + stds + energies + diffs + temporal

        if config.model_name == "cnn":
            return means + stds + energies
        if config.model_name == "gru":
            return (
                means + diffs + [window[-1][i] - window[0][i] for i in range(channels)]
            )
        if config.model_name in {"bilstm", "bilstmattention"}:
            forward_delta = [window[-1][i] - window[0][i] for i in range(channels)]
            backward_delta = [window[0][i] - window[-1][i] for i in range(channels)]
            features = means + stds + diffs + temporal + forward_delta + backward_delta
            if config.model_name == "bilstmattention":
                features += attention + spectral[:channels]
            return features
        if config.model_name == "tiny_transformer":
            midpoint = len(window) // 2
            first_half = [
                sum(window[i][c] for i in range(midpoint)) / max(midpoint, 1)
                for c in range(channels)
            ]
            second_half = [
                sum(window[i][c] for i in range(midpoint, len(window)))
                / max(len(window) - midpoint, 1)
                for c in range(channels)
            ]
            return (
                means
                + stds
                + temporal
                + [second_half[i] - first_half[i] for i in range(channels)]
                + spectral[:channels]
            )
        if config.model_name == "deepconvlstm":
            return (
                means + stds + energies + diffs + temporal + multiscale + interactions
            )
        if config.model_name == "deepconvlstmattention":
            return (
                means
                + stds
                + energies
                + attention
                + multiscale
                + interactions
                + spectral[:channels]
            )
        if config.model_name == "hart":
            return (
                means
                + stds
                + saliency
                + temporal
                + multiscale
                + interactions
                + spectral[:channels]
            )
        if config.model_name == "tinierhar":
            return means + stds + saliency + temporal + interactions
        if config.model_name == "tinyhar":
            return means + stds + temporal + spectral[:channels]
        if config.model_name == "spectra":
            return means + stds + spectral + diffs
        if config.model_name == "cnn_gru":
            return means + stds + energies + diffs + temporal + interactions
        if config.model_name == "custom":
            base_features = (
                default_features + spectral[:channels] + interactions + multiscale
            )
            return self._custom_model_features(
                window=window,
                base_features=base_features,
                params=config.model_params,
            )
        return default_features

    def _custom_model_features(
        self,
        window: list[list[float]],
        base_features: list[float],
        params: dict[str, float | int | str | bool],
    ) -> list[float]:
        custom_code = str(params.get("custom_code", "")).strip()
        if not custom_code:
            return list(base_features)
        feature_fn = self._custom_model_feature_callable(custom_code)
        if not callable(feature_fn):
            return list(base_features)
        try:
            result = feature_fn(
                self._copy_window(window),
                list(base_features),
                dict(self._safe_custom_params(params)),
            )
        except Exception:  # noqa: BLE001
            return list(base_features)
        return self._validated_custom_features(result, base_features)

    def _custom_model_feature_callable(self, custom_code: str) -> object | None:
        return self._cached_custom_callable(
            cache=self._custom_model_feature_cache,
            custom_code=custom_code,
            function_name="custom_model_features",
        )

    def _validated_custom_features(
        self,
        candidate: object,
        fallback: list[float],
    ) -> list[float]:
        if not isinstance(candidate, list):
            return list(fallback)
        max_feature_count = max(len(fallback) * 4, 16)
        if not 1 <= len(candidate) <= max_feature_count:
            return list(fallback)
        clean_features: list[float] = []
        for value in candidate:
            if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                return list(fallback)
            clean_features.append(float(value))
        return clean_features

    def _train_model(
        self,
        model_name: str,
        model_params: dict[str, float | int | str | bool],
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
                model_params=model_params,
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
        if model_name in {
            "deepconvlstm",
            "deepconvlstmattention",
            "hart",
            "bilstm",
            "bilstmattention",
            "tinierhar",
            "tinyhar",
            "spectra",
            "custom",
        }:
            hidden_dim = int(
                model_params.get(
                    "hidden_dim",
                    48
                    if model_name in {"deepconvlstm", "deepconvlstmattention", "hart"}
                    or model_name in {"bilstm", "bilstmattention"}
                    else 24,
                )
            )
            depth = int(
                model_params.get("depth", 2 if model_name != "tinierhar" else 1)
            )
            return self._train_mlp_model(
                train_features=train_features,
                train_labels=train_labels,
                stop_features=stop_features,
                stop_labels=stop_labels,
                learning_rate=learning_rate,
                epochs=epochs,
                hidden_dim=hidden_dim,
                depth=depth,
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
        lr_scale = 1.0 / math.sqrt(feature_dim)
        base_lr = max(0.005, min(learning_rate * 400.0, 0.2)) * lr_scale
        order = list(range(len(train_features)))
        best_weights = [row[:] for row in weights]
        best_score = float("-inf")
        best_epoch = 0
        stale_epochs = 0
        current_lr = base_lr
        grad_clip = 2.0

        for epoch in range(max(epochs * 4, 8)):
            if epoch > 0 and epoch % 8 == 0 and stale_epochs > 0:
                current_lr *= 0.92
            random.Random(epochs + epoch).shuffle(order)
            for sample_index in order:
                features = train_features[sample_index]
                label = train_labels[sample_index]
                logits = [self._dot(weight, features) for weight in weights]
                probs = self._softmax(logits)
                for class_index in range(num_classes):
                    error = probs[class_index] - (1.0 if class_index == label else 0.0)
                    for feature_index, value in enumerate(features):
                        grad = current_lr * error * value
                        grad = max(-grad_clip, min(grad_clip, grad))
                        weights[class_index][feature_index] -= grad
                    bias_grad = current_lr * error
                    bias_grad = max(-grad_clip, min(grad_clip, bias_grad))
                    weights[class_index][-1] -= bias_grad
            stop_predictions = [
                self._predict({"kind": "linear", "weights": weights}, features)
                for features in stop_features
            ]
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
        depth: int,
        patience: int,
        min_epochs: int,
    ) -> tuple[dict[str, object], dict[str, object]]:
        feature_dim = len(train_features[0])
        num_classes = len(self.LABELS)
        rng = random.Random(feature_dim + hidden_dim + epochs)
        fan_in = feature_dim
        scale = math.sqrt(2.0 / (fan_in + hidden_dim))
        hidden_layers: list[list[list[float]]] = []
        input_dim = feature_dim
        for layer_idx in range(max(1, depth)):
            scale = math.sqrt(2.0 / (input_dim + hidden_dim))
            hidden_layers.append(
                [
                    [rng.gauss(0.0, scale) for _ in range(input_dim + 1)]
                    for _ in range(hidden_dim)
                ]
            )
            input_dim = hidden_dim
        output_scale = math.sqrt(2.0 / (hidden_dim + num_classes))
        output = [
            [rng.gauss(0.0, output_scale) for _ in range(hidden_dim + 1)]
            for _ in range(num_classes)
        ]
        lr_scale = 1.0 / math.sqrt(feature_dim)
        base_lr = max(0.001, min(learning_rate * 50.0, 0.05)) * lr_scale
        order = list(range(len(train_features)))
        best_hidden_layers = [[row[:] for row in layer] for layer in hidden_layers]
        best_output = [row[:] for row in output]
        best_score = float("-inf")
        best_epoch = 0
        stale_epochs = 0
        current_lr = base_lr
        grad_clip = 5.0
        for epoch in range(max(epochs * 3, 6)):
            if epoch > 0 and epoch % 10 == 0 and stale_epochs > 0:
                current_lr *= 0.9
            random.Random(hidden_dim + epoch).shuffle(order)
            for sample_index in order:
                features = train_features[sample_index]
                label = train_labels[sample_index]
                layer_inputs = [features]
                layer_states: list[list[float]] = []
                for layer in hidden_layers:
                    hidden_linear = [
                        self._dot(weight, layer_inputs[-1]) for weight in layer
                    ]
                    hidden_state = [math.tanh(value) for value in hidden_linear]
                    layer_states.append(hidden_state)
                    layer_inputs.append(hidden_state)
                logits = [self._dot(weight, hidden_state) for weight in output]
                probs = self._softmax(logits)
                output_errors = [
                    probs[class_index] - (1.0 if class_index == label else 0.0)
                    for class_index in range(num_classes)
                ]
                downstream_errors: list[list[float]] = []
                last_hidden = layer_states[-1]
                last_errors = []
                for hidden_index in range(hidden_dim):
                    downstream = sum(
                        output[class_index][hidden_index] * output_errors[class_index]
                        for class_index in range(num_classes)
                    )
                    grad = (1.0 - (last_hidden[hidden_index] ** 2)) * downstream
                    grad = max(-grad_clip, min(grad_clip, grad))
                    last_errors.append(grad)
                downstream_errors.append(last_errors)
                grad_norms: list[float] = []
                for class_index in range(num_classes):
                    grad_norm = output_errors[class_index] * current_lr
                    grad_norm = max(-grad_clip, min(grad_norm, grad_clip))
                    for hidden_index in range(hidden_dim):
                        output[class_index][hidden_index] -= (
                            grad_norm * last_hidden[hidden_index]
                        )
                    output[class_index][-1] -= grad_norm
                    grad_norms.append(abs(grad_norm))
                for layer_index in range(len(hidden_layers) - 2, -1, -1):
                    layer_state = layer_states[layer_index]
                    next_layer = hidden_layers[layer_index + 1]
                    next_errors = downstream_errors[0]
                    current_errors = []
                    for hidden_index in range(hidden_dim):
                        downstream = sum(
                            next_layer[next_index][hidden_index]
                            * next_errors[next_index]
                            for next_index in range(hidden_dim)
                        )
                        grad = (1.0 - (layer_state[hidden_index] ** 2)) * downstream
                        grad = max(-grad_clip, min(grad_clip, grad))
                        current_errors.append(grad)
                    downstream_errors.insert(0, current_errors)
                for layer_index, layer in enumerate(hidden_layers):
                    errors = downstream_errors[layer_index]
                    inputs = layer_inputs[layer_index]
                    max_grad = 0.0
                    for hidden_index in range(hidden_dim):
                        for feature_index in range(len(inputs)):
                            g = (
                                current_lr
                                * errors[hidden_index]
                                * inputs[feature_index]
                            )
                            max_grad = max(max_grad, abs(g))
                            layer[hidden_index][feature_index] -= g
                        bias_grad = current_lr * errors[hidden_index]
                        layer[hidden_index][-1] -= bias_grad
                        max_grad = max(max_grad, abs(bias_grad))
            stop_model = {
                "kind": "mlp",
                "hidden_layers": hidden_layers,
                "output": output,
            }
            stop_predictions = [
                self._predict(stop_model, features) for features in stop_features
            ]
            stop_score = self._macro_f1(stop_predictions, stop_labels)
            if stop_score > best_score:
                best_score = stop_score
                best_epoch = epoch + 1
                stale_epochs = 0
                best_hidden_layers = [
                    [row[:] for row in layer] for layer in hidden_layers
                ]
                best_output = [row[:] for row in output]
            else:
                stale_epochs += 1
            if (epoch + 1) >= min_epochs and stale_epochs >= patience:
                break
        return (
            {"kind": "mlp", "hidden_layers": best_hidden_layers, "output": best_output},
            {
                "device": "cpu",
                "epochs_trained": epoch + 1,
                "best_epoch": best_epoch,
                "best_stop_macro_f1": round(best_score, 4),
                "stopped_early": stale_epochs >= patience,
                "final_lr": round(current_lr, 6),
            },
        )

    def _train_torch_model(
        self,
        model_name: str,
        model_params: dict[str, float | int | str | bool],
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
        module = self._build_architecture_from_file(
            model_name=model_name,
            feature_dim=feature_dim,
            num_classes=num_classes,
            model_params=model_params,
        )
        if module is not None:
            module = module.to(device)
            optimizer = self._build_torch_optimizer(
                module=module,
                model_params=model_params,
                learning_rate=learning_rate,
            )
            criterion = self._build_torch_criterion(model_params)
            return self._fit_torch_module(
                module=module,
                optimizer=optimizer,
                criterion=criterion,
                train_features=train_features,
                train_labels=train_labels,
                stop_features=stop_features,
                stop_labels=stop_labels,
                epochs=epochs,
                patience=patience,
                min_epochs=min_epochs,
                device=device,
            )
        if model_name in {
            "deepconvlstm",
            "deepconvlstmattention",
            "hart",
            "bilstm",
            "bilstmattention",
            "tinierhar",
            "tinyhar",
            "spectra",
            "custom",
        }:
            hidden_dim = int(
                model_params.get(
                    "hidden_dim",
                    64
                    if model_name in {"deepconvlstm", "deepconvlstmattention", "hart"}
                    or model_name in {"bilstm", "bilstmattention"}
                    else 32,
                )
            )
            depth = max(
                1, int(model_params.get("depth", 2 if model_name != "tinierhar" else 1))
            )
            if model_name == "deepconvlstmattention":
                module = self._build_torch_attention_mlp(
                    feature_dim=feature_dim,
                    hidden_dim=hidden_dim,
                    depth=depth,
                    num_classes=num_classes,
                    attention_mix=float(model_params.get("attention_mix", 0.8)),
                )
            elif model_name in {"bilstm", "bilstmattention"}:
                module = self._build_torch_attention_mlp(
                    feature_dim=feature_dim,
                    hidden_dim=hidden_dim * 2
                    if bool(model_params.get("bidirectional", True))
                    else hidden_dim,
                    depth=depth,
                    num_classes=num_classes,
                    attention_mix=float(
                        model_params.get(
                            "attention_mix",
                            0.7 if model_name == "bilstmattention" else 0.35,
                        )
                    ),
                )
            elif model_name in {
                "deepconvlstm",
                "hart",
                "tinierhar",
                "tinyhar",
                "spectra",
                "custom",
            }:
                layers: list[object] = []
                input_dim = feature_dim
                for _ in range(depth):
                    layers.append(self._torch.nn.Linear(input_dim, hidden_dim))
                    layers.append(self._torch.nn.ReLU())
                    input_dim = hidden_dim
                layers.append(self._torch.nn.Linear(input_dim, num_classes))
                module = self._torch.nn.Sequential(*layers)
        else:
            module = self._torch.nn.Linear(feature_dim, num_classes)
        module = module.to(device)
        optimizer = self._build_torch_optimizer(
            module=module,
            model_params=model_params,
            learning_rate=learning_rate,
        )
        criterion = self._build_torch_criterion(model_params)
        return self._fit_torch_module(
            module=module,
            optimizer=optimizer,
            criterion=criterion,
            train_features=train_features,
            train_labels=train_labels,
            stop_features=stop_features,
            stop_labels=stop_labels,
            epochs=epochs,
            patience=patience,
            min_epochs=min_epochs,
            device=device,
        )

    def _build_architecture_from_file(
        self,
        model_name: str,
        feature_dim: int,
        num_classes: int,
        model_params: dict[str, float | int | str | bool],
    ) -> object | None:
        assert self._torch is not None
        safe_name = model_name.replace("-", "_")
        try:
            module = importlib.import_module(
                f"research.modules.architectures.{safe_name}"
            )
        except Exception:
            return None
        builder = getattr(module, "build_model", None)
        if not callable(builder):
            return None
        try:
            candidate = builder(
                input_dim=feature_dim,
                num_classes=num_classes,
                params=dict(model_params),
            )
        except Exception:
            return None
        return candidate if isinstance(candidate, self._torch.nn.Module) else None

    def _fit_torch_module(
        self,
        module: object,
        optimizer: object,
        criterion: object,
        train_features: list[list[float]],
        train_labels: list[int],
        stop_features: list[list[float]],
        stop_labels: list[int],
        epochs: int,
        patience: int,
        min_epochs: int,
        device: object,
    ) -> tuple[dict[str, object], dict[str, object]]:
        assert self._torch is not None
        train_x = self._torch.tensor(
            train_features, dtype=self._torch.float32, device=device
        )
        train_y = self._torch.tensor(
            train_labels, dtype=self._torch.long, device=device
        )
        stop_x = self._torch.tensor(
            stop_features, dtype=self._torch.float32, device=device
        )
        batch_size = min(128, len(train_features))
        best_state = {
            key: value.detach().cpu().clone()
            for key, value in module.state_dict().items()
        }
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
                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in module.state_dict().items()
                }
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

    def _build_torch_optimizer(
        self,
        module: object,
        model_params: dict[str, float | int | str | bool],
        learning_rate: float,
    ) -> object:
        assert self._torch is not None
        optimizer_name = str(model_params.get("optimizer", "adam")).lower()
        weight_decay = max(0.0, float(model_params.get("weight_decay", 0.0)))
        lr = max(learning_rate, 1e-4)
        if optimizer_name == "sgd":
            return self._torch.optim.SGD(
                module.parameters(),
                lr=lr,
                momentum=float(model_params.get("momentum", 0.9)),
                weight_decay=weight_decay,
            )
        if optimizer_name == "adamw":
            return self._torch.optim.AdamW(
                module.parameters(), lr=lr, weight_decay=weight_decay
            )
        return self._torch.optim.Adam(
            module.parameters(), lr=lr, weight_decay=weight_decay
        )

    def _build_torch_criterion(
        self,
        model_params: dict[str, float | int | str | bool],
    ) -> object:
        assert self._torch is not None
        loss_name = str(model_params.get("loss", "cross_entropy")).lower()
        label_smoothing = max(
            0.0, min(0.2, float(model_params.get("label_smoothing", 0.0)))
        )
        if loss_name in {"cross_entropy", "weighted_cross_entropy"}:
            try:
                return self._torch.nn.CrossEntropyLoss(
                    label_smoothing=label_smoothing
                )
            except TypeError:
                return self._torch.nn.CrossEntropyLoss()
        try:
            return self._torch.nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        except TypeError:
            return self._torch.nn.CrossEntropyLoss()

    def _predict(self, model: dict[str, object], features: list[float]) -> int:
        if model["kind"] == "linear":
            logits = [self._dot(weight, features) for weight in model["weights"]]
        elif model["kind"] == "torch":
            assert self._torch is not None
            with self._torch.no_grad():
                tensor = self._torch.tensor(
                    features, dtype=self._torch.float32
                ).unsqueeze(0)
                logits = model["module"](tensor).squeeze(0).tolist()
        else:
            hidden_state = features
            for layer in model["hidden_layers"]:
                hidden_state = [
                    math.tanh(self._dot(weight, hidden_state)) for weight in layer
                ]
            logits = [self._dot(weight, hidden_state) for weight in model["output"]]
        return max(range(len(logits)), key=lambda index: logits[index])

    def _parameter_count(self, model: dict[str, object]) -> int:
        if model["kind"] == "linear":
            return len(model["weights"]) * len(model["weights"][0])
        if model["kind"] == "torch":
            return sum(parameter.numel() for parameter in model["module"].parameters())
        hidden_layers = model["hidden_layers"]
        output = model["output"]
        hidden_params = sum(len(layer) * len(layer[0]) for layer in hidden_layers)
        return hidden_params + (len(output) * len(output[0]))

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

    def _temporal_pool_features(
        self, window: list[list[float]], mode: str
    ) -> list[float]:
        channels = len(window[0])
        if mode == "attention":
            return self._attention_pool_features(window, 0.75)
        if mode == "last":
            return window[-1][:]
        midpoint = len(window) // 2
        first_half = [
            sum(window[i][c] for i in range(midpoint)) / max(midpoint, 1)
            for c in range(channels)
        ]
        second_half = [
            sum(window[i][c] for i in range(midpoint, len(window)))
            / max(len(window) - midpoint, 1)
            for c in range(channels)
        ]
        if mode == "concat":
            return first_half + second_half
        return [0.5 * (first_half[c] + second_half[c]) for c in range(channels)]

    def _attention_pool_features(
        self, window: list[list[float]], attention_mix: float
    ) -> list[float]:
        channels = len(window[0])
        energies = [sum(value * value for value in row) for row in window]
        temperature = max(1e-4, 1.0 - (0.5 * attention_mix))
        scaled = [energy / temperature for energy in energies]
        max_scaled = max(scaled)
        weights = [math.exp(value - max_scaled) for value in scaled]
        normalizer = sum(weights) + 1e-8
        normalized = [weight / normalizer for weight in weights]
        attention = [
            sum(weight * row[channel] for weight, row in zip(normalized, window))
            for channel in range(channels)
        ]
        mean_pool = [
            sum(row[channel] for row in window) / len(window)
            for channel in range(channels)
        ]
        return [
            ((1.0 - attention_mix) * mean_pool[channel])
            + (attention_mix * attention[channel])
            for channel in range(channels)
        ]

    def _spectral_features(
        self, window: list[list[float]], spectral_bins: int
    ) -> list[float]:
        channels = len(window[0])
        bins = max(2, min(spectral_bins, max(2, len(window) // 8)))
        features: list[float] = []
        for channel in range(channels):
            signal = [row[channel] for row in window]
            channel_features: list[float] = []
            for k in range(1, bins + 1):
                real = 0.0
                imag = 0.0
                for index, value in enumerate(signal):
                    angle = (2.0 * math.pi * k * index) / len(signal)
                    real += value * math.cos(angle)
                    imag -= value * math.sin(angle)
                channel_features.append(
                    math.sqrt((real * real) + (imag * imag)) / len(signal)
                )
            features.extend(channel_features)
        return features

    def _build_torch_attention_mlp(
        self,
        feature_dim: int,
        hidden_dim: int,
        depth: int,
        num_classes: int,
        attention_mix: float,
    ):
        assert self._torch is not None
        torch = self._torch
        mix = max(0.0, min(1.0, float(attention_mix)))

        class AttentionMLP(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.backbone = torch.nn.ModuleList()
                input_dim = feature_dim
                for _ in range(max(1, depth)):
                    self.backbone.append(torch.nn.Linear(input_dim, hidden_dim))
                    input_dim = hidden_dim
                self.attention = torch.nn.Linear(hidden_dim, hidden_dim)
                self.classifier = torch.nn.Linear(hidden_dim, num_classes)
                self.mix = mix

            def forward(self, x):
                h = x
                for layer in self.backbone:
                    h = torch.relu(layer(h))
                gates = torch.sigmoid(self.attention(h))
                pooled = ((1.0 - self.mix) * h) + (self.mix * (h * gates))
                return self.classifier(pooled)

        return AttentionMLP()

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
            sum(
                abs(window[index][channel] - window[index - 1][channel])
                for channel in range(channels)
            )
            for index in range(1, len(window))
        ]
        if not motion:
            return [0.0 for _ in range(channels * 2)]
        top_indices = sorted(
            range(len(motion)), key=lambda index: motion[index], reverse=True
        )[: max(2, len(motion) // 8)]
        salient_rows = [window[index + 1] for index in top_indices]
        pooled = [
            sum(row[channel] for row in salient_rows) / len(salient_rows)
            for channel in range(channels)
        ]
        peak = [
            max(abs(row[channel]) for row in salient_rows)
            for channel in range(channels)
        ]
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
        return (
            sum(coef * value for coef, value in zip(weight[:-1], features)) + weight[-1]
        )

    def _softmax(self, logits: list[float]) -> list[float]:
        max_logit = max(logits)
        exps = [math.exp(logit - max_logit) for logit in logits]
        total = sum(exps)
        return [value / total for value in exps]

    def _accuracy(self, predictions: list[int], labels: list[int]) -> float:
        correct = sum(
            int(prediction == label) for prediction, label in zip(predictions, labels)
        )
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

    def _per_class_f1(
        self, predictions: list[int], labels: list[int]
    ) -> dict[str, float]:
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
            score = (
                0.0
                if precision + recall == 0
                else (2.0 * precision * recall) / (precision + recall)
            )
            scores[label_name] = round(score, 4)
        return scores

    def _confusion_matrix(
        self, predictions: list[int], labels: list[int]
    ) -> list[list[int]]:
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
                    "model_params": config.model_params,
                    "preprocessing": config.preprocessing,
                    "data_params": config.data_params,
                    "augmentation": config.augmentation,
                    "augmentation_strength": config.augmentation_strength,
                    "augmentation_params": config.augmentation_params,
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
                "model_params": config.model_params,
                "preprocessing": config.preprocessing,
                "data_params": config.data_params,
                "augmentation": config.augmentation,
                "augmentation_strength": config.augmentation_strength,
                "augmentation_params": config.augmentation_params,
                "learning_rate": config.learning_rate,
                "window_size": config.window_size,
            },
            "training_report": dict(training_report),
        }
        if model["kind"] == "linear":
            payload["weights"] = model["weights"]
        elif model["kind"] == "mlp":
            payload["hidden_layers"] = model["hidden_layers"]
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
