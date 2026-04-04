from __future__ import annotations

import json
import pprint
from pathlib import Path

from autohar_researcher.types import (
    AugmentationProposal,
    CandidateConfig,
    DataProposal,
    ModelProposal,
)

MODULE_FILENAMES = {
    "model_agent": "model_module.py",
    "augmentation_agent": "augmentation_module.py",
    "data_agent": "data_module.py",
}


def default_model_params(name: str) -> dict[str, float | int | str | bool]:
    params: dict[str, float | int | str | bool] = {
        "hidden_dim": 64,
        "depth": 2,
        "temporal_pool": "avg",
        "spectral_bins": 6,
        "use_interactions": True,
        "use_multiscale": True,
        "attention_mix": 0.5,
    }
    if name == "cnn":
        params.update({"hidden_dim": 32, "depth": 1, "use_interactions": False})
    elif name == "gru":
        params.update({"hidden_dim": 40, "depth": 1, "temporal_pool": "last"})
    elif name == "tiny_transformer":
        params.update({"hidden_dim": 48, "depth": 2, "temporal_pool": "concat"})
    elif name == "hart":
        params.update({"hidden_dim": 72, "depth": 2, "temporal_pool": "concat"})
    elif name == "deepconvlstmattention":
        params.update(
            {
                "hidden_dim": 80,
                "depth": 2,
                "temporal_pool": "attention",
                "spectral_bins": 8,
                "attention_mix": 0.8,
            }
        )
    elif name == "tinierhar":
        params.update({"hidden_dim": 32, "depth": 1, "temporal_pool": "avg", "use_multiscale": False})
    elif name == "tinyhar":
        params.update({"hidden_dim": 40, "depth": 1, "temporal_pool": "avg", "spectral_bins": 4})
    elif name == "spectra":
        params.update(
            {
                "hidden_dim": 56,
                "depth": 2,
                "temporal_pool": "concat",
                "spectral_bins": 10,
                "use_interactions": False,
            }
        )
    elif name == "cnn_gru":
        params.update({"hidden_dim": 56, "depth": 2, "temporal_pool": "last"})
    return params


def default_augmentation_params(name: str, strength: str) -> dict[str, float | int]:
    strength_key = strength if strength in {"weak", "medium", "strong"} else "medium"
    strength_scale = {
        "weak": 0.5,
        "medium": 1.0,
        "strong": 1.5,
    }[strength_key]
    if name == "rotation":
        return {"angle_deg": round(10.0 * strength_scale, 3)}
    if name == "jitter":
        return {"jitter_std": round(0.02 * strength_scale, 5)}
    if name == "scaling":
        return {"scale_span": round(0.10 * strength_scale, 5)}
    if name == "time_warp":
        return {"compression_ratio": round(max(0.7, 1.0 - (0.08 * strength_scale)), 5)}
    if name == "permutation":
        return {"segment_count": int(min(6, max(3, round(2 + (2 * strength_scale)))))}
    if name == "drift":
        return {"drift_max": round(0.015 * strength_scale, 5)}
    if name == "magnitude_warp":
        return {"warp_span": round(0.15 * strength_scale, 5)}
    if name == "mixup":
        return {"blend": round(min(0.45, 0.15 + (0.15 * strength_scale)), 5)}
    return {}


def default_data_params(preprocessing: str, window_size: int) -> dict[str, float | int | str | bool]:
    params: dict[str, float | int | str | bool] = {
        "channel_layout": "total_acc+gyro",
        "resample_mode": "nearest",
        "normalize": preprocessing in {"normalized", "normalized+gravity_filter", "normalized+magnitude_channels"},
        "gravity_scale": 1.0,
        "append_magnitude": preprocessing == "normalized+magnitude_channels",
    }
    if preprocessing == "normalized+gravity_filter":
        params["channel_layout"] = "body_acc+gyro"
    if window_size >= 192:
        params["resample_mode"] = "linear"
    return params


def base_experiment_config() -> dict[str, object]:
    return {
        "dataset": {
            "name": "uci_har",
            "split": "official_train_test",
        },
        "model": {
            "name": "deepconvlstm",
            "learning_rate": 3e-4,
            "params": default_model_params("deepconvlstm"),
        },
        "data": {
            "preprocessing": "raw",
            "window_size": 128,
            "params": default_data_params("raw", 128),
        },
        "augmentation": {
            "name": "none",
            "strength": "medium",
            "params": default_augmentation_params("none", "medium"),
        },
        "training": {
            "epochs": 12,
            "min_epochs": 4,
            "early_stopping_patience": 3,
            "device": "auto",
            "metric": "macro_f1",
        },
    }


def module_relative_path(agent_name: str) -> Path:
    return Path("research") / "modules" / MODULE_FILENAMES[agent_name]


def module_owned_fields(agent_name: str) -> list[str]:
    return {
        "model_agent": ["model.name", "model.learning_rate", "model.params"],
        "augmentation_agent": ["augmentation.name", "augmentation.strength", "augmentation.params"],
        "data_agent": ["data.preprocessing", "data.window_size", "data.params"],
    }[agent_name]


def module_config_from_experiment(
    agent_name: str,
    experiment: dict[str, object],
) -> dict[str, object]:
    if agent_name == "model_agent":
        return {
            "name": experiment["model"]["name"],
            "learning_rate": experiment["model"]["learning_rate"],
            "params": dict(experiment["model"].get("params", {})),
            "notes": experiment["model"].get("notes", ""),
        }
    if agent_name == "augmentation_agent":
        return {
            "name": experiment["augmentation"]["name"],
            "strength": experiment["augmentation"]["strength"],
            "params": dict(experiment["augmentation"].get("params", {})),
            "notes": experiment["augmentation"].get("notes", ""),
        }
    return {
        "preprocessing": experiment["data"]["preprocessing"],
        "window_size": experiment["data"]["window_size"],
        "params": dict(experiment["data"].get("params", {})),
        "notes": experiment["data"].get("notes", ""),
    }


def module_payload(
    agent_name: str,
    experiment: dict[str, object],
    *,
    round_id: int,
    branch_name: str,
    sync_from_branch: str,
    logic_expression: str = "",
    helper_code: str = "",
    function_rewrite: str = "",
) -> dict[str, object]:
    return {
        "round_id": round_id,
        "agent_name": agent_name,
        "branch_name": branch_name,
        "sync_from_branch": sync_from_branch,
        "owned_fields": module_owned_fields(agent_name),
        "config": module_config_from_experiment(agent_name, experiment),
        "logic_expression": logic_expression,
        "helper_code": helper_code,
        "function_rewrite": function_rewrite,
    }


def write_module_python(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    config_body = pprint.pformat(payload["config"], sort_dicts=True, width=88)
    owned_fields_body = pprint.pformat(payload["owned_fields"], sort_dicts=True, width=88)
    patch_builder = _module_patch_builder(payload["agent_name"])
    path.write_text(
        '"""Agent-owned executable module for AutoHARResearcher."""\n\n'
        f'AGENT_NAME = {payload["agent_name"]!r}\n'
        f'BRANCH_NAME = {payload["branch_name"]!r}\n'
        f"ROUND_ID = {int(payload['round_id'])}\n"
        f'SYNC_FROM_BRANCH = {payload["sync_from_branch"]!r}\n'
        f"OWNED_FIELDS = {owned_fields_body}\n"
        f"CONFIG = {config_body}\n\n"
        f"LOGIC_EXPRESSION = {payload.get('logic_expression', '')!r}\n\n"
        f"LLM_HELPER_CODE = {payload.get('helper_code', '')!r}\n\n"
        f"LLM_FUNCTION_REWRITE = {payload.get('function_rewrite', '')!r}\n\n"
        "def module_metadata() -> dict[str, object]:\n"
        '    """Return serializable metadata for this owned module."""\n'
        "    return {\n"
        '        "agent_name": AGENT_NAME,\n'
        '        "branch_name": BRANCH_NAME,\n'
        '        "round_id": ROUND_ID,\n'
        '        "sync_from_branch": SYNC_FROM_BRANCH,\n'
        '        "owned_fields": list(OWNED_FIELDS),\n'
        '        "config": dict(CONFIG),\n'
        '        "logic_expression": LOGIC_EXPRESSION,\n'
        '        "helper_code": LLM_HELPER_CODE,\n'
        '        "function_rewrite": LLM_FUNCTION_REWRITE,\n'
        "    }\n\n"
        f"{patch_builder}\n",
        encoding="utf-8",
    )


def _module_patch_builder(agent_name: str) -> str:
    if agent_name == "model_agent":
        return (
            "def current_model(base_experiment: dict[str, object]) -> dict[str, object]:\n"
            '    """Return the current baseline model section."""\n'
            '    return dict(base_experiment.get("model", {}))\n\n'
            "def stable_model_families() -> list[str]:\n"
            '    """Return model families that are usually safe early choices."""\n'
            '    return ["deepconvlstm", "deepconvlstmattention", "cnn_gru", "gru", "cnn", "hart", "tinierhar", "tinyhar", "spectra", "tiny_transformer"]\n\n'
            "def choose_target_name(requested_name: str, baseline_name: str) -> str:\n"
            '    """Prefer valid, stable model families and fall back safely."""\n'
            "    valid = stable_model_families()\n"
            "    if requested_name in valid:\n"
            "        return requested_name\n"
            "    if baseline_name in valid:\n"
            "        return baseline_name\n"
            '    return valid[0]\n\n'
            "def compatibility_bias(base_experiment: dict[str, object], target_name: str) -> float:\n"
            '    """Return a small LR bias based on the surrounding baseline context."""\n'
            '    augmentation = str(base_experiment.get("augmentation", {}).get("name", "none"))\n'
            '    preprocessing = str(base_experiment.get("data", {}).get("preprocessing", "raw"))\n'
            '    bias = 1.0\n'
            '    if augmentation != "none" and target_name in {"deepconvlstm", "deepconvlstmattention", "hart", "tiny_transformer"}:\n'
            "        bias *= 0.9\n"
            '    if preprocessing != "raw" and target_name in {"tiny_transformer", "hart"}:\n'
            "        bias *= 0.95\n"
            "    return bias\n\n"
            "def bounded_learning_rate(target_name: str, baseline_name: str, baseline_lr: float, requested_lr: float) -> float:\n"
            '    """Keep LR changes conservative relative to the active baseline."""\n'
            "    if target_name == baseline_name:\n"
            "        lower = max(1e-4, baseline_lr * 0.5)\n"
            "        upper = min(5e-3, baseline_lr * 1.5)\n"
            "        return max(lower, min(upper, requested_lr))\n"
            "    return max(1e-4, min(5e-3, requested_lr))\n\n"
            "def default_model_params(target_name: str) -> dict[str, float | int | str | bool]:\n"
            '    """Return concrete model-side parameters for the chosen family."""\n'
            "    params = {\n"
            '        "hidden_dim": 64,\n'
            '        "depth": 2,\n'
            '        "temporal_pool": "avg",\n'
            '        "spectral_bins": 6,\n'
            '        "use_interactions": True,\n'
            '        "use_multiscale": True,\n'
            '        "attention_mix": 0.5,\n'
            "    }\n"
            '    if target_name == "cnn":\n'
            '        params.update({"hidden_dim": 32, "depth": 1, "use_interactions": False})\n'
            '    elif target_name == "gru":\n'
            '        params.update({"hidden_dim": 40, "depth": 1, "temporal_pool": "last"})\n'
            '    elif target_name == "tiny_transformer":\n'
            '        params.update({"hidden_dim": 48, "depth": 2, "temporal_pool": "concat"})\n'
            '    elif target_name == "hart":\n'
            '        params.update({"hidden_dim": 72, "depth": 2, "temporal_pool": "concat"})\n'
            '    elif target_name == "deepconvlstmattention":\n'
            '        params.update({"hidden_dim": 80, "depth": 2, "temporal_pool": "attention", "spectral_bins": 8, "attention_mix": 0.8})\n'
            '    elif target_name == "tinierhar":\n'
            '        params.update({"hidden_dim": 32, "depth": 1, "temporal_pool": "avg", "use_multiscale": False})\n'
            '    elif target_name == "tinyhar":\n'
            '        params.update({"hidden_dim": 40, "depth": 1, "temporal_pool": "avg", "spectral_bins": 4})\n'
            '    elif target_name == "spectra":\n'
            '        params.update({"hidden_dim": 56, "depth": 2, "temporal_pool": "concat", "spectral_bins": 10, "use_interactions": False})\n'
            '    elif target_name == "cnn_gru":\n'
            '        params.update({"hidden_dim": 56, "depth": 2, "temporal_pool": "last"})\n'
            "    return params\n\n"
            "def strategy_summary(base_experiment: dict[str, object]) -> str:\n"
            '    """Describe the model-side mutation relative to baseline."""\n'
            "    baseline = current_model(base_experiment)\n"
            '    baseline_name = str(baseline.get("name", "deepconvlstm"))\n'
            '    baseline_lr = float(baseline.get("learning_rate", 3e-4))\n'
            '    baseline_params = dict(baseline.get("params", {}))\n'
            '    return f"baseline_model={baseline_name}, baseline_lr={baseline_lr}, baseline_params={baseline_params}"\n\n'
            "def allowed_builtins() -> dict[str, object]:\n"
            '    """Return a small builtins set for LLM helper execution."""\n'
            "    return {\n"
            '        "abs": abs,\n'
            '        "all": all,\n'
            '        "any": any,\n'
            '        "dict": dict,\n'
            '        "float": float,\n'
            '        "int": int,\n'
            '        "len": len,\n'
            '        "list": list,\n'
            '        "max": max,\n'
            '        "min": min,\n'
            '        "round": round,\n'
            '        "set": set,\n'
            '        "sorted": sorted,\n'
            '        "str": str,\n'
            '        "sum": sum,\n'
            '        "tuple": tuple,\n'
            "    }\n\n"
            "def llm_helper_namespace() -> dict[str, object]:\n"
            '    """Load LLM-provided helper code inside a constrained namespace."""\n'
            '    if not LLM_HELPER_CODE:\n'
            "        return {}\n"
            '    namespace = {"__builtins__": allowed_builtins()}\n'
            '    try:\n'
            '        exec(LLM_HELPER_CODE, namespace, namespace)\n'
            '    except Exception:\n'
            "        return {}\n"
            "    return namespace\n\n"
            "def llm_function_namespace() -> dict[str, object]:\n"
            '    """Load a bounded LLM function rewrite inside a constrained namespace."""\n'
            '    if not LLM_FUNCTION_REWRITE:\n'
            "        return {}\n"
            '    namespace = {"__builtins__": allowed_builtins()}\n'
            '    try:\n'
            '        exec(LLM_FUNCTION_REWRITE, namespace, namespace)\n'
            '    except Exception:\n'
            "        return {}\n"
            "    return namespace\n\n"
            "def llm_bounded_learning_rate_override(base_experiment: dict[str, object], target_name: str, baseline_name: str, baseline_lr: float, proposed_lr: float) -> float | None:\n"
            '    """Run a bounded rewrite of LR bounding if provided."""\n'
            "    namespace = llm_function_namespace()\n"
            '    chooser = namespace.get("bounded_learning_rate_override")\n'
            '    if not callable(chooser):\n'
            "        return None\n"
            '    config = dict(CONFIG)\n'
            '    try:\n'
            '        result = chooser(base_experiment, target_name, baseline_name, baseline_lr, proposed_lr, config)\n'
            '    except Exception:\n'
            "        return None\n"
            '    if isinstance(result, (int, float)):\n'
            '        return float(result)\n'
            "    return None\n\n"
            "def llm_model_params_override(base_experiment: dict[str, object], target_name: str, proposed_params: dict[str, object]) -> dict[str, object] | None:\n"
            '    """Run a bounded rewrite of model-parameter compatibility if provided."""\n'
            "    namespace = llm_function_namespace()\n"
            '    chooser = namespace.get("model_params_override")\n'
            '    if not callable(chooser):\n'
            "        return None\n"
            '    config = dict(CONFIG)\n'
            '    try:\n'
            '        result = chooser(base_experiment, target_name, dict(proposed_params), config)\n'
            '    except Exception:\n'
            "        return None\n"
            '    return result if isinstance(result, dict) else None\n\n'
            "def llm_helper_adjustments(base_experiment: dict[str, object]) -> dict[str, object]:\n"
            '    """Run LLM-owned helper code if it defines llm_choose_patch."""\n'
            "    namespace = llm_helper_namespace()\n"
            '    chooser = namespace.get("llm_choose_patch")\n'
            '    if not callable(chooser):\n'
            "        return {}\n"
            '    config = dict(CONFIG)\n'
            '    try:\n'
            '        result = chooser(base_experiment, config)\n'
            '    except Exception:\n'
            "        return {}\n"
            "    return result if isinstance(result, dict) else {}\n\n"
            "def llm_adjustments(base_experiment: dict[str, object]) -> dict[str, object]:\n"
            '    """Apply planner-provided local logic for model-side overrides."""\n'
            '    helper_result = llm_helper_adjustments(base_experiment)\n'
            '    if helper_result:\n'
            "        return helper_result\n"
            '    if not LOGIC_EXPRESSION:\n'
            "        return {}\n"
            '    config = dict(CONFIG)\n'
            '    try:\n'
            '        result = eval(LOGIC_EXPRESSION, {"__builtins__": allowed_builtins()}, {"base_experiment": base_experiment, "config": config})\n'
            '    except Exception:\n'
            "        return {}\n"
            '    if callable(result):\n'
            '        try:\n'
            '            result = result(base_experiment, config)\n'
            '        except Exception:\n'
            "            return {}\n"
            "    return result if isinstance(result, dict) else {}\n\n"
            "def build_patch(base_experiment: dict[str, object]) -> dict[str, object]:\n"
            '    """Return the model-only patch owned by ModelAgent."""\n'
            '    baseline_model = current_model(base_experiment)\n'
            '    baseline_name = str(baseline_model.get("name", "deepconvlstm"))\n'
            '    baseline_lr = float(baseline_model.get("learning_rate", 3e-4))\n'
            '    target_name = choose_target_name(str(CONFIG.get("name", baseline_name)), baseline_name)\n'
            '    proposed_lr = float(CONFIG.get("learning_rate", baseline_lr))\n'
            '    proposed_lr *= compatibility_bias(base_experiment, target_name)\n'
            '    proposed_lr = bounded_learning_rate(target_name, baseline_name, baseline_lr, proposed_lr)\n'
            '    rewritten_lr = llm_bounded_learning_rate_override(base_experiment, target_name, baseline_name, baseline_lr, proposed_lr)\n'
            '    if rewritten_lr is not None:\n'
            '        proposed_lr = max(1e-4, min(5e-3, rewritten_lr))\n'
            '    target_params = default_model_params(target_name)\n'
            '    if isinstance(CONFIG.get("params"), dict):\n'
            '        target_params.update(CONFIG["params"])\n'
            '    rewritten_params = llm_model_params_override(base_experiment, target_name, target_params)\n'
            '    if isinstance(rewritten_params, dict):\n'
            '        target_params.update(rewritten_params)\n'
            "    patch = {\n"
            '        "model": {\n'
            '            "name": target_name,\n'
            '            "learning_rate": proposed_lr,\n'
            '            "params": target_params,\n'
            '            "notes": f\"{CONFIG.get(\'notes\', \'\')} | {strategy_summary(base_experiment)}\",\n'
            "        }\n"
            "    }\n"
            '    adjustments = llm_adjustments(base_experiment)\n'
            '    if isinstance(adjustments.get("model"), dict):\n'
            '        if "model_name" in adjustments["model"] and "name" not in adjustments["model"]:\n'
            '            adjustments["model"]["name"] = adjustments["model"].pop("model_name")\n'
            '        if isinstance(adjustments["model"].get("params"), dict):\n'
            '            merged_params = dict(patch["model"].get("params", {}))\n'
            '            merged_params.update(adjustments["model"]["params"])\n'
            '            adjustments["model"]["params"] = merged_params\n'
            '        patch["model"].update(adjustments["model"])\n'
            "    return patch\n"
        )
    if agent_name == "augmentation_agent":
        return (
            "def current_augmentation(base_experiment: dict[str, object]) -> dict[str, object]:\n"
            '    """Return the current baseline augmentation section."""\n'
            '    return dict(base_experiment.get("augmentation", {}))\n\n'
            "def valid_augmentations() -> list[str]:\n"
            '    """Return supported augmentation families."""\n'
            '    return ["none", "rotation", "time_warp", "jitter", "mixup", "scaling", "drift", "permutation", "magnitude_warp"]\n\n'
            "def choose_augmentation_name(base_experiment: dict[str, object], requested_name: str) -> str:\n"
            '    """Apply simple compatibility heuristics for augmentation selection."""\n'
            "    valid = valid_augmentations()\n"
            "    if requested_name not in valid:\n"
            '        requested_name = str(current_augmentation(base_experiment).get("name", "none"))\n'
            '    model_name = str(base_experiment.get("model", {}).get("name", "deepconvlstm"))\n'
            '    if requested_name == "mixup" and model_name in {"gru", "cnn"}:\n'
            '        return "rotation"\n'
            "    return requested_name\n\n"
            "def normalize_strength(target_name: str, requested_strength: str) -> str:\n"
            '    """Keep augmentation strength in the supported regime."""\n'
            '    valid_strengths = {"weak", "medium", "strong"}\n'
            '    strength = requested_strength if requested_strength in valid_strengths else "medium"\n'
            '    if target_name == "none":\n'
            '        return "medium"\n'
            '    return strength\n\n'
            "def default_augmentation_params(target_name: str, target_strength: str) -> dict[str, float | int]:\n"
            '    """Return concrete augmentation parameters for the chosen family and strength."""\n'
            '    strength_key = target_strength if target_strength in {"weak", "medium", "strong"} else "medium"\n'
            '    strength_scale = {"weak": 0.5, "medium": 1.0, "strong": 1.5}[strength_key]\n'
            '    if target_name == "rotation":\n'
            '        return {"angle_deg": round(10.0 * strength_scale, 3)}\n'
            '    if target_name == "jitter":\n'
            '        return {"jitter_std": round(0.02 * strength_scale, 5)}\n'
            '    if target_name == "scaling":\n'
            '        return {"scale_span": round(0.10 * strength_scale, 5)}\n'
            '    if target_name == "time_warp":\n'
            '        return {"compression_ratio": round(max(0.7, 1.0 - (0.08 * strength_scale)), 5)}\n'
            '    if target_name == "permutation":\n'
            '        return {"segment_count": int(min(6, max(3, round(2 + (2 * strength_scale)))))}\n'
            '    if target_name == "drift":\n'
            '        return {"drift_max": round(0.015 * strength_scale, 5)}\n'
            '    if target_name == "magnitude_warp":\n'
            '        return {"warp_span": round(0.15 * strength_scale, 5)}\n'
            '    if target_name == "mixup":\n'
            '        return {"blend": round(min(0.45, 0.15 + (0.15 * strength_scale)), 5)}\n'
            '    return {}\n\n'
            "def compatibility_strength(base_experiment: dict[str, object], target_name: str, requested_strength: str) -> str:\n"
            '    """Downshift aggressive augmentation when the baseline context is fragile."""\n'
            '    strength = normalize_strength(target_name, requested_strength)\n'
            '    model_name = str(base_experiment.get("model", {}).get("name", "deepconvlstm"))\n'
            '    preprocessing = str(base_experiment.get("data", {}).get("preprocessing", "raw"))\n'
            '    if target_name in {"time_warp", "permutation", "magnitude_warp"} and strength == "strong":\n'
            '        return "medium"\n'
            '    if preprocessing != "raw" and model_name in {"tiny_transformer", "hart"} and strength == "strong":\n'
            '        return "medium"\n'
            "    return strength\n\n"
            "def strategy_summary(base_experiment: dict[str, object]) -> str:\n"
            '    """Describe the augmentation-side mutation relative to baseline."""\n'
            "    baseline = current_augmentation(base_experiment)\n"
            '    baseline_name = str(baseline.get("name", "none"))\n'
            '    baseline_strength = str(baseline.get("strength", "medium"))\n'
            '    baseline_params = dict(baseline.get("params", {}))\n'
            '    return f"baseline_augmentation={baseline_name}:{baseline_strength}, baseline_params={baseline_params}"\n\n'
            "def allowed_builtins() -> dict[str, object]:\n"
            '    """Return a small builtins set for LLM helper execution."""\n'
            "    return {\n"
            '        "abs": abs,\n'
            '        "all": all,\n'
            '        "any": any,\n'
            '        "dict": dict,\n'
            '        "float": float,\n'
            '        "int": int,\n'
            '        "len": len,\n'
            '        "list": list,\n'
            '        "max": max,\n'
            '        "min": min,\n'
            '        "round": round,\n'
            '        "set": set,\n'
            '        "sorted": sorted,\n'
            '        "str": str,\n'
            '        "sum": sum,\n'
            '        "tuple": tuple,\n'
            "    }\n\n"
            "def llm_helper_namespace() -> dict[str, object]:\n"
            '    """Load LLM-provided helper code inside a constrained namespace."""\n'
            '    if not LLM_HELPER_CODE:\n'
            "        return {}\n"
            '    namespace = {"__builtins__": allowed_builtins()}\n'
            '    try:\n'
            '        exec(LLM_HELPER_CODE, namespace, namespace)\n'
            '    except Exception:\n'
            "        return {}\n"
            "    return namespace\n\n"
            "def llm_function_namespace() -> dict[str, object]:\n"
            '    """Load a bounded LLM function rewrite inside a constrained namespace."""\n'
            '    if not LLM_FUNCTION_REWRITE:\n'
            "        return {}\n"
            '    namespace = {"__builtins__": allowed_builtins()}\n'
            '    try:\n'
            '        exec(LLM_FUNCTION_REWRITE, namespace, namespace)\n'
            '    except Exception:\n'
            "        return {}\n"
            "    return namespace\n\n"
            "def llm_compatibility_strength_override(base_experiment: dict[str, object], target_name: str, requested_strength: str) -> str | None:\n"
            '    """Run a bounded rewrite of augmentation strength compatibility if provided."""\n'
            "    namespace = llm_function_namespace()\n"
            '    chooser = namespace.get("compatibility_strength_override")\n'
            '    if not callable(chooser):\n'
            "        return None\n"
            '    config = dict(CONFIG)\n'
            '    try:\n'
            '        result = chooser(base_experiment, target_name, requested_strength, config)\n'
            '    except Exception:\n'
            "        return None\n"
            '    if isinstance(result, str) and result in {"weak", "medium", "strong"}:\n'
            "        return result\n"
            "    return None\n\n"
            "def llm_helper_adjustments(base_experiment: dict[str, object]) -> dict[str, object]:\n"
            '    """Run LLM-owned helper code if it defines llm_choose_patch."""\n'
            "    namespace = llm_helper_namespace()\n"
            '    chooser = namespace.get("llm_choose_patch")\n'
            '    if not callable(chooser):\n'
            "        return {}\n"
            '    config = dict(CONFIG)\n'
            '    try:\n'
            '        result = chooser(base_experiment, config)\n'
            '    except Exception:\n'
            "        return {}\n"
            "    return result if isinstance(result, dict) else {}\n\n"
            "def llm_adjustments(base_experiment: dict[str, object]) -> dict[str, object]:\n"
            '    """Apply planner-provided local logic for augmentation-side overrides."""\n'
            '    helper_result = llm_helper_adjustments(base_experiment)\n'
            '    if helper_result:\n'
            "        return helper_result\n"
            '    if not LOGIC_EXPRESSION:\n'
            "        return {}\n"
            '    config = dict(CONFIG)\n'
            '    try:\n'
            '        result = eval(LOGIC_EXPRESSION, {"__builtins__": allowed_builtins()}, {"base_experiment": base_experiment, "config": config})\n'
            '    except Exception:\n'
            "        return {}\n"
            '    if callable(result):\n'
            '        try:\n'
            '            result = result(base_experiment, config)\n'
            '        except Exception:\n'
            "            return {}\n"
            "    return result if isinstance(result, dict) else {}\n\n"
            "def build_patch(base_experiment: dict[str, object]) -> dict[str, object]:\n"
            '    """Return the augmentation-only patch owned by AugmentationAgent."""\n'
            '    baseline_aug = current_augmentation(base_experiment)\n'
            '    target_name = choose_augmentation_name(base_experiment, str(CONFIG.get("name", baseline_aug.get("name", "none"))))\n'
            '    target_strength = str(CONFIG.get("strength", baseline_aug.get("strength", "medium")))\n'
            '    target_strength = compatibility_strength(base_experiment, target_name, target_strength)\n'
            '    rewritten_strength = llm_compatibility_strength_override(base_experiment, target_name, target_strength)\n'
            '    if rewritten_strength is not None:\n'
            '        target_strength = rewritten_strength\n'
            '    target_params = default_augmentation_params(target_name, target_strength)\n'
            '    if isinstance(CONFIG.get("params"), dict):\n'
            '        target_params.update(CONFIG["params"])\n'
            "    patch = {\n"
            '        "augmentation": {\n'
            '            "name": target_name,\n'
            '            "strength": target_strength,\n'
            '            "params": target_params,\n'
            '            "notes": f\"{CONFIG.get(\'notes\', \'\')} | {strategy_summary(base_experiment)}\",\n'
            "        }\n"
            "    }\n"
            '    adjustments = llm_adjustments(base_experiment)\n'
            '    if isinstance(adjustments.get("augmentation"), dict):\n'
            '        if "type" in adjustments["augmentation"] and "name" not in adjustments["augmentation"]:\n'
            '            adjustments["augmentation"]["name"] = adjustments["augmentation"].pop("type")\n'
            '        if isinstance(adjustments["augmentation"].get("params"), dict):\n'
            '            merged_params = dict(patch["augmentation"].get("params", {}))\n'
            '            merged_params.update(adjustments["augmentation"]["params"])\n'
            '            adjustments["augmentation"]["params"] = merged_params\n'
            '        patch["augmentation"].update(adjustments["augmentation"])\n'
            "    return patch\n"
        )
    return (
        "def current_data(base_experiment: dict[str, object]) -> dict[str, object]:\n"
        '    """Return the current baseline data section."""\n'
        '    return dict(base_experiment.get("data", {}))\n\n'
        "def valid_preprocessing_modes() -> list[str]:\n"
        '    """Return supported preprocessing modes."""\n'
        '    return ["raw", "normalized", "normalized+gravity_filter", "normalized+magnitude_channels"]\n\n'
        "def nearest_valid_window(requested_window: int) -> int:\n"
        '    """Snap a requested window size to the supported search space."""\n'
        "    valid_windows = [64, 96, 128, 192, 256]\n"
        "    return min(valid_windows, key=lambda candidate: (abs(candidate - requested_window), candidate))\n\n"
        "def choose_preprocessing(base_experiment: dict[str, object], requested_mode: str) -> str:\n"
        '    """Apply simple model-aware preprocessing preferences."""\n'
        "    valid = valid_preprocessing_modes()\n"
        "    if requested_mode not in valid:\n"
        '        requested_mode = str(current_data(base_experiment).get("preprocessing", "raw"))\n'
        '    model_name = str(base_experiment.get("model", {}).get("name", "deepconvlstm"))\n'
        '    if requested_mode == "normalized+gravity_filter" and model_name in {"tiny_transformer", "hart"}:\n'
        '        return "normalized"\n'
        "    return requested_mode\n\n"
        "def choose_window(base_experiment: dict[str, object], requested_window: int) -> int:\n"
        '    """Apply simple compatibility heuristics for window size."""\n'
        "    snapped = nearest_valid_window(requested_window)\n"
        '    augmentation = str(base_experiment.get("augmentation", {}).get("name", "none"))\n'
        '    if augmentation in {"time_warp", "permutation"} and snapped > 192:\n'
        "        return 192\n"
        "    return snapped\n\n"
        "def default_data_params(target_preprocessing: str, target_window: int) -> dict[str, float | int | str | bool]:\n"
        '    """Return concrete data pipeline parameters for the chosen preprocessing/window."""\n'
        "    params = {\n"
        '        "channel_layout": "total_acc+gyro",\n'
        '        "resample_mode": "nearest",\n'
        '        "normalize": target_preprocessing in {"normalized", "normalized+gravity_filter", "normalized+magnitude_channels"},\n'
        '        "gravity_scale": 1.0,\n'
        '        "append_magnitude": target_preprocessing == "normalized+magnitude_channels",\n'
        "    }\n"
        '    if target_preprocessing == "normalized+gravity_filter":\n'
        '        params["channel_layout"] = "body_acc+gyro"\n'
        "    if target_window >= 192:\n"
        '        params["resample_mode"] = "linear"\n'
        "    return params\n\n"
        "def strategy_summary(base_experiment: dict[str, object]) -> str:\n"
        '    """Describe the data-side mutation relative to baseline."""\n'
        "    baseline = current_data(base_experiment)\n"
        '    baseline_pre = str(baseline.get("preprocessing", "raw"))\n'
        '    baseline_window = int(baseline.get("window_size", 128))\n'
        '    baseline_params = dict(baseline.get("params", {}))\n'
        '    return f"baseline_data={baseline_pre}, baseline_window={baseline_window}, baseline_params={baseline_params}"\n\n'
        "def allowed_builtins() -> dict[str, object]:\n"
        '    """Return a small builtins set for LLM helper execution."""\n'
        "    return {\n"
        '        "abs": abs,\n'
        '        "all": all,\n'
        '        "any": any,\n'
        '        "dict": dict,\n'
        '        "float": float,\n'
        '        "int": int,\n'
        '        "len": len,\n'
        '        "list": list,\n'
        '        "max": max,\n'
        '        "min": min,\n'
        '        "round": round,\n'
        '        "set": set,\n'
        '        "sorted": sorted,\n'
        '        "str": str,\n'
        '        "sum": sum,\n'
        '        "tuple": tuple,\n'
        "    }\n\n"
        "def llm_helper_namespace() -> dict[str, object]:\n"
        '    """Load LLM-provided helper code inside a constrained namespace."""\n'
        '    if not LLM_HELPER_CODE:\n'
        "        return {}\n"
        '    namespace = {"__builtins__": allowed_builtins()}\n'
        '    try:\n'
        '        exec(LLM_HELPER_CODE, namespace, namespace)\n'
        '    except Exception:\n'
        "        return {}\n"
        "    return namespace\n\n"
        "def llm_function_namespace() -> dict[str, object]:\n"
        '    """Load a bounded LLM function rewrite inside a constrained namespace."""\n'
        '    if not LLM_FUNCTION_REWRITE:\n'
        "        return {}\n"
        '    namespace = {"__builtins__": allowed_builtins()}\n'
        '    try:\n'
        '        exec(LLM_FUNCTION_REWRITE, namespace, namespace)\n'
        '    except Exception:\n'
        "        return {}\n"
        "    return namespace\n\n"
        "def llm_choose_window_override(base_experiment: dict[str, object], requested_window: int) -> int | None:\n"
        '    """Run a bounded rewrite of data window selection if provided."""\n'
        "    namespace = llm_function_namespace()\n"
        '    chooser = namespace.get("choose_window_override")\n'
        '    if not callable(chooser):\n'
        "        return None\n"
        '    config = dict(CONFIG)\n'
        '    try:\n'
        '        result = chooser(base_experiment, requested_window, config)\n'
        '    except Exception:\n'
        "        return None\n"
        '    if isinstance(result, int):\n'
        "        return result\n"
        "    return None\n\n"
        "def llm_helper_adjustments(base_experiment: dict[str, object]) -> dict[str, object]:\n"
        '    """Run LLM-owned helper code if it defines llm_choose_patch."""\n'
        "    namespace = llm_helper_namespace()\n"
        '    chooser = namespace.get("llm_choose_patch")\n'
        '    if not callable(chooser):\n'
        "        return {}\n"
        '    config = dict(CONFIG)\n'
        '    try:\n'
        '        result = chooser(base_experiment, config)\n'
        '    except Exception:\n'
        "        return {}\n"
        "    return result if isinstance(result, dict) else {}\n\n"
        "def llm_adjustments(base_experiment: dict[str, object]) -> dict[str, object]:\n"
        '    """Apply planner-provided local logic for data-side overrides."""\n'
        '    helper_result = llm_helper_adjustments(base_experiment)\n'
        '    if helper_result:\n'
        "        return helper_result\n"
        '    if not LOGIC_EXPRESSION:\n'
        "        return {}\n"
        '    config = dict(CONFIG)\n'
        '    try:\n'
        '        result = eval(LOGIC_EXPRESSION, {"__builtins__": allowed_builtins()}, {"base_experiment": base_experiment, "config": config})\n'
        '    except Exception:\n'
        "        return {}\n"
        '    if callable(result):\n'
        '        try:\n'
        '            result = result(base_experiment, config)\n'
        '        except Exception:\n'
        "            return {}\n"
        "    return result if isinstance(result, dict) else {}\n\n"
        "def build_patch(base_experiment: dict[str, object]) -> dict[str, object]:\n"
        '    """Return the data-only patch owned by DataAgent."""\n'
        '    baseline_data = current_data(base_experiment)\n'
        '    target_preprocessing = choose_preprocessing(base_experiment, str(CONFIG.get("preprocessing", baseline_data.get("preprocessing", "raw"))))\n'
        '    requested_window = int(CONFIG.get("window_size", baseline_data.get("window_size", 128)))\n'
        '    target_window = choose_window(base_experiment, requested_window)\n'
        '    rewritten_window = llm_choose_window_override(base_experiment, requested_window)\n'
        '    if rewritten_window is not None:\n'
        '        target_window = nearest_valid_window(rewritten_window)\n'
        '    target_params = default_data_params(target_preprocessing, target_window)\n'
        '    if isinstance(CONFIG.get("params"), dict):\n'
        '        target_params.update(CONFIG["params"])\n'
        "    patch = {\n"
        '        "data": {\n'
        '            "preprocessing": target_preprocessing,\n'
        '            "window_size": target_window,\n'
        '            "params": target_params,\n'
        '            "notes": f\"{CONFIG.get(\'notes\', \'\')} | {strategy_summary(base_experiment)}\",\n'
        "        }\n"
        "    }\n"
        '    adjustments = llm_adjustments(base_experiment)\n'
        '    if isinstance(adjustments.get("data"), dict):\n'
        '        if isinstance(adjustments["data"].get("params"), dict):\n'
        '            merged_params = dict(patch["data"].get("params", {}))\n'
        '            merged_params.update(adjustments["data"]["params"])\n'
        '            adjustments["data"]["params"] = merged_params\n'
        '        patch["data"].update(adjustments["data"])\n'
        "    return patch\n"
    )


def load_module_python(path: Path) -> dict[str, object]:
    namespace: dict[str, object] = {}
    exec(path.read_text(encoding="utf-8"), namespace, namespace)
    module_metadata = namespace.get("module_metadata")
    if callable(module_metadata):
        payload = module_metadata()
    else:
        payload = namespace.get("MODULE_PAYLOAD", {})
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid module payload in {path}")
    return payload


def apply_module_python(
    experiment: dict[str, object],
    path: Path,
) -> tuple[dict[str, object], dict[str, object]]:
    namespace: dict[str, object] = {}
    exec(path.read_text(encoding="utf-8"), namespace, namespace)
    module_metadata = namespace.get("module_metadata")
    build_patch = namespace.get("build_patch")
    if callable(module_metadata) and callable(build_patch):
        payload = module_metadata()
        patch = build_patch(clone_experiment(experiment))
    else:
        payload = namespace.get("MODULE_PAYLOAD", {})
        if not isinstance(payload, dict):
            raise ValueError(f"Invalid executable module in {path}")
        config = payload.get("config", {})
        agent_name = payload.get("agent_name")
        if agent_name == "model_agent":
            patch = {
                "model": {
                    "name": str(config["name"]),
                    "learning_rate": float(config["learning_rate"]),
                    "params": dict(config.get("params", {})),
                    "notes": str(config.get("notes", "")),
                }
            }
        elif agent_name == "augmentation_agent":
            patch = {
                "augmentation": {
                    "name": str(config["name"]),
                    "strength": str(config["strength"]),
                    "params": dict(config.get("params", {})),
                    "notes": str(config.get("notes", "")),
                }
            }
        else:
            patch = {
                "data": {
                    "preprocessing": str(config["preprocessing"]),
                    "window_size": int(config["window_size"]),
                    "params": dict(config.get("params", {})),
                    "notes": str(config.get("notes", "")),
                }
            }
    if not isinstance(payload, dict) or not isinstance(patch, dict):
        raise ValueError(f"Invalid module return values in {path}")
    updated = clone_experiment(experiment)
    for section_name, section_patch in patch.items():
        if isinstance(section_patch, dict) and section_name in updated:
            updated[section_name].update(section_patch)
        else:
            updated[section_name] = section_patch
    return updated, payload


def compose_experiment_from_workspace(
    worktree_path: Path,
    fallback_experiment: dict[str, object] | None = None,
) -> dict[str, object]:
    experiment_path = worktree_path / "research" / "experiment.json"
    if experiment_path.exists():
        payload = json.loads(experiment_path.read_text(encoding="utf-8"))
        if "experiment" in payload:
            experiment = clone_experiment(payload["experiment"])
        else:
            experiment = clone_experiment(payload)
    elif fallback_experiment is not None:
        experiment = clone_experiment(fallback_experiment)
    else:
        experiment = base_experiment_config()

    for agent_name in MODULE_FILENAMES:
        module_path = worktree_path / module_relative_path(agent_name)
        if not module_path.exists():
            continue
        experiment, _ = apply_module_python(experiment, module_path)
    return experiment


def apply_model_proposal(
    experiment: dict[str, object],
    proposal: ModelProposal,
) -> dict[str, object]:
    updated = clone_experiment(experiment)
    updated["model"]["name"] = proposal.model_name
    updated["model"]["learning_rate"] = proposal.learning_rate
    updated["model"]["params"] = dict(proposal.model_params) if proposal.model_params else default_model_params(proposal.model_name)
    updated["model"]["notes"] = proposal.notes
    return updated


def apply_augmentation_proposal(
    experiment: dict[str, object],
    proposal: AugmentationProposal,
) -> dict[str, object]:
    updated = clone_experiment(experiment)
    updated["augmentation"]["name"] = proposal.augmentation
    updated["augmentation"]["strength"] = proposal.strength
    updated["augmentation"]["params"] = default_augmentation_params(
        proposal.augmentation,
        proposal.strength,
    )
    updated["augmentation"]["notes"] = proposal.notes
    return updated


def apply_data_proposal(
    experiment: dict[str, object],
    proposal: DataProposal,
) -> dict[str, object]:
    updated = clone_experiment(experiment)
    updated["data"]["preprocessing"] = proposal.preprocessing
    updated["data"]["window_size"] = proposal.window_size
    updated["data"]["params"] = default_data_params(
        proposal.preprocessing,
        proposal.window_size,
    )
    updated["data"]["notes"] = proposal.notes
    return updated


def experiment_to_candidate_config(experiment: dict[str, object], notes: str) -> CandidateConfig:
    return CandidateConfig(
        model_name=str(experiment["model"]["name"]),
        model_params=dict(experiment["model"].get("params", {})),
        preprocessing=str(experiment["data"]["preprocessing"]),
        data_params=dict(experiment["data"].get("params", {})),
        augmentation=str(experiment["augmentation"]["name"]),
        augmentation_strength=str(experiment["augmentation"].get("strength", "medium")),
        learning_rate=float(experiment["model"]["learning_rate"]),
        window_size=int(experiment["data"]["window_size"]),
        augmentation_params=dict(experiment["augmentation"].get("params", {})),
        notes=notes,
    )


def clone_experiment(experiment: dict[str, object]) -> dict[str, object]:
    return {
        "dataset": dict(experiment["dataset"]),
        "model": {
            **dict(experiment["model"]),
            "params": dict(experiment["model"].get("params", {})),
        },
        "data": {
            **dict(experiment["data"]),
            "params": dict(experiment["data"].get("params", {})),
        },
        "augmentation": {
            **dict(experiment["augmentation"]),
            "params": dict(experiment["augmentation"].get("params", {})),
        },
        "training": dict(experiment["training"]),
    }
