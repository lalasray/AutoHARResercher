from __future__ import annotations


def file_patches_schema() -> dict[str, object]:
    return {
        "type": "array",
        "items": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "file_path": {"type": "string"},
                "content": {"type": "string"},
                "patch_mode": {"type": "string"},
                "rationale": {"type": "string"},
            },
            "required": ["file_path", "content", "patch_mode", "rationale"],
        },
    }


def orchestrator_plan_schema() -> dict[str, object]:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "summary": {"type": "string"},
            "priority_order": {
                "type": "array",
                "items": {"type": "string"},
                "minItems": 3,
                "maxItems": 3,
            },
            "exploration_mode": {"type": "string"},
            "combination_bias": {"type": "string"},
            "risk_note": {"type": "string"},
        },
        "required": [
            "summary",
            "priority_order",
            "exploration_mode",
            "combination_bias",
            "risk_note",
        ],
    }


def model_agent_plan_schema() -> dict[str, object]:
    model_param_properties = {
        "hidden_dim": {"type": "integer"},
        "depth": {"type": "integer"},
        "temporal_pool": {"type": "string"},
        "spectral_bins": {"type": "integer"},
        "use_interactions": {"type": "boolean"},
        "use_multiscale": {"type": "boolean"},
        "attention_mix": {"type": "number"},
        "dropout": {"type": "number"},
        "bidirectional": {"type": "boolean"},
        "optimizer": {"type": "string"},
        "weight_decay": {"type": "number"},
        "loss": {"type": "string"},
        "label_smoothing": {"type": "number"},
        "parent_architectures": {"type": "string"},
        "merge_strategy": {"type": "string"},
    }
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "model_name": {"type": "string"},
            "learning_rate": {"type": "number"},
            "model_params": {
                "type": "object",
                "additionalProperties": False,
                "properties": model_param_properties,
                "required": list(model_param_properties),
            },
            "rationale": {"type": "string"},
            "logic_expression": {"type": "string"},
            "helper_code": {"type": "string"},
            "function_rewrite": {"type": "string"},
            "file_patches": file_patches_schema(),
        },
        "required": [
            "model_name",
            "learning_rate",
            "model_params",
            "rationale",
            "logic_expression",
            "helper_code",
            "function_rewrite",
            "file_patches",
        ],
    }


def augmentation_agent_plan_schema() -> dict[str, object]:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "augmentation": {"type": "string"},
            "strength": {"type": "string"},
            "rationale": {"type": "string"},
            "logic_expression": {"type": "string"},
            "helper_code": {"type": "string"},
            "function_rewrite": {"type": "string"},
            "file_patches": file_patches_schema(),
        },
        "required": [
            "augmentation",
            "strength",
            "rationale",
            "logic_expression",
            "helper_code",
            "function_rewrite",
            "file_patches",
        ],
    }


def data_params_schema() -> dict[str, object]:
    nullable_scalar = {"type": ["string", "number", "integer", "boolean", "null"]}
    properties = {
        "channel_layout": nullable_scalar,
        "normalization": nullable_scalar,
        "resample_hz": nullable_scalar,
        "lowpass_cutoff": nullable_scalar,
        "gravity_filter_order": nullable_scalar,
        "include_magnitude": nullable_scalar,
        "stride": nullable_scalar,
        "dataset_policy": nullable_scalar,
        "train_datasets": nullable_scalar,
        "evaluation_dataset": nullable_scalar,
        "dataset_sources": nullable_scalar,
        "real_ratio": nullable_scalar,
        "per_dataset_real_ratio": nullable_scalar,
        "synthetic_ratio": nullable_scalar,
        "synthetic_sources": nullable_scalar,
        "synthetic_target_classes": nullable_scalar,
        "synthetic_source_targets": nullable_scalar,
        "per_source_synthetic_ratio": nullable_scalar,
        "class_sampling_policy": nullable_scalar,
        "target_classes": nullable_scalar,
        "per_class_real_ratio": nullable_scalar,
        "per_dataset_class_real_ratio": nullable_scalar,
        "per_class_synthetic_ratio": nullable_scalar,
    }
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": properties,
        "required": list(properties),
    }


def data_agent_plan_schema() -> dict[str, object]:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "preprocessing": {"type": "string"},
            "window_size": {"type": "integer"},
            "data_params": data_params_schema(),
            "rationale": {"type": "string"},
            "logic_expression": {"type": "string"},
            "helper_code": {"type": "string"},
            "function_rewrite": {"type": "string"},
            "file_patches": file_patches_schema(),
        },
        "required": [
            "preprocessing",
            "window_size",
            "data_params",
            "rationale",
            "logic_expression",
            "helper_code",
            "function_rewrite",
            "file_patches",
        ],
    }


def proposal_judgment_schema() -> dict[str, object]:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "mutation_size": {"type": "string"},
            "novelty": {"type": "string"},
            "plausibility": {"type": "string"},
            "compatibility": {"type": "string"},
            "evaluate_now": {"type": "boolean"},
            "rationale": {"type": "string"},
        },
        "required": [
            "mutation_size",
            "novelty",
            "plausibility",
            "compatibility",
            "evaluate_now",
            "rationale",
        ],
    }


def proposal_judgments_schema() -> dict[str, object]:
    judgment_schema = proposal_judgment_schema()
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "model_agent": judgment_schema,
            "augmentation_agent": judgment_schema,
            "data_agent": judgment_schema,
        },
        "required": ["model_agent", "augmentation_agent", "data_agent"],
    }
