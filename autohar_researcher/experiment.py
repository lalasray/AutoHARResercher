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


def base_experiment_config() -> dict[str, object]:
    return {
        "dataset": {
            "name": "uci_har",
            "split": "official_train_test",
        },
        "model": {
            "name": "deepconvlstm",
            "learning_rate": 3e-4,
        },
        "data": {
            "preprocessing": "raw",
            "window_size": 128,
        },
        "augmentation": {
            "name": "none",
            "strength": "medium",
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
        "model_agent": ["model.name", "model.learning_rate"],
        "augmentation_agent": ["augmentation.name", "augmentation.strength"],
        "data_agent": ["data.preprocessing", "data.window_size"],
    }[agent_name]


def module_config_from_experiment(
    agent_name: str,
    experiment: dict[str, object],
) -> dict[str, object]:
    if agent_name == "model_agent":
        return {
            "name": experiment["model"]["name"],
            "learning_rate": experiment["model"]["learning_rate"],
            "notes": experiment["model"].get("notes", ""),
        }
    if agent_name == "augmentation_agent":
        return {
            "name": experiment["augmentation"]["name"],
            "strength": experiment["augmentation"]["strength"],
            "notes": experiment["augmentation"].get("notes", ""),
        }
    return {
        "preprocessing": experiment["data"]["preprocessing"],
        "window_size": experiment["data"]["window_size"],
        "notes": experiment["data"].get("notes", ""),
    }


def module_payload(
    agent_name: str,
    experiment: dict[str, object],
    *,
    round_id: int,
    branch_name: str,
    sync_from_branch: str,
) -> dict[str, object]:
    return {
        "round_id": round_id,
        "agent_name": agent_name,
        "branch_name": branch_name,
        "sync_from_branch": sync_from_branch,
        "owned_fields": module_owned_fields(agent_name),
        "config": module_config_from_experiment(agent_name, experiment),
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
        "def module_metadata() -> dict[str, object]:\n"
        '    """Return serializable metadata for this owned module."""\n'
        "    return {\n"
        '        "agent_name": AGENT_NAME,\n'
        '        "branch_name": BRANCH_NAME,\n'
        '        "round_id": ROUND_ID,\n'
        '        "sync_from_branch": SYNC_FROM_BRANCH,\n'
        '        "owned_fields": list(OWNED_FIELDS),\n'
        '        "config": dict(CONFIG),\n'
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
            '    return ["deepconvlstm", "cnn_gru", "gru", "cnn", "hart", "tinierhar", "tiny_transformer"]\n\n'
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
            '    if augmentation != "none" and target_name in {"deepconvlstm", "hart", "tiny_transformer"}:\n'
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
            "def strategy_summary(base_experiment: dict[str, object]) -> str:\n"
            '    """Describe the model-side mutation relative to baseline."""\n'
            "    baseline = current_model(base_experiment)\n"
            '    baseline_name = str(baseline.get("name", "deepconvlstm"))\n'
            '    baseline_lr = float(baseline.get("learning_rate", 3e-4))\n'
            '    return f"baseline_model={baseline_name}, baseline_lr={baseline_lr}"\n\n'
            "def build_patch(base_experiment: dict[str, object]) -> dict[str, object]:\n"
            '    """Return the model-only patch owned by ModelAgent."""\n'
            '    baseline_model = current_model(base_experiment)\n'
            '    baseline_name = str(baseline_model.get("name", "deepconvlstm"))\n'
            '    baseline_lr = float(baseline_model.get("learning_rate", 3e-4))\n'
            '    target_name = choose_target_name(str(CONFIG.get("name", baseline_name)), baseline_name)\n'
            '    proposed_lr = float(CONFIG.get("learning_rate", baseline_lr))\n'
            '    proposed_lr *= compatibility_bias(base_experiment, target_name)\n'
            '    proposed_lr = bounded_learning_rate(target_name, baseline_name, baseline_lr, proposed_lr)\n'
            "    return {\n"
            '        "model": {\n'
            '            "name": target_name,\n'
            '            "learning_rate": proposed_lr,\n'
            '            "notes": f\"{CONFIG.get(\'notes\', \'\')} | {strategy_summary(base_experiment)}\",\n'
            "        }\n"
            "    }\n"
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
            '    return f"baseline_augmentation={baseline_name}:{baseline_strength}"\n\n'
            "def build_patch(base_experiment: dict[str, object]) -> dict[str, object]:\n"
            '    """Return the augmentation-only patch owned by AugmentationAgent."""\n'
            '    baseline_aug = current_augmentation(base_experiment)\n'
            '    target_name = choose_augmentation_name(base_experiment, str(CONFIG.get("name", baseline_aug.get("name", "none"))))\n'
            '    target_strength = str(CONFIG.get("strength", baseline_aug.get("strength", "medium")))\n'
            '    target_strength = compatibility_strength(base_experiment, target_name, target_strength)\n'
            "    return {\n"
            '        "augmentation": {\n'
            '            "name": target_name,\n'
            '            "strength": target_strength,\n'
            '            "notes": f\"{CONFIG.get(\'notes\', \'\')} | {strategy_summary(base_experiment)}\",\n'
            "        }\n"
            "    }\n"
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
        "def strategy_summary(base_experiment: dict[str, object]) -> str:\n"
        '    """Describe the data-side mutation relative to baseline."""\n'
        "    baseline = current_data(base_experiment)\n"
        '    baseline_pre = str(baseline.get("preprocessing", "raw"))\n'
        '    baseline_window = int(baseline.get("window_size", 128))\n'
        '    return f"baseline_data={baseline_pre}, baseline_window={baseline_window}"\n\n'
        "def build_patch(base_experiment: dict[str, object]) -> dict[str, object]:\n"
        '    """Return the data-only patch owned by DataAgent."""\n'
        '    baseline_data = current_data(base_experiment)\n'
        '    target_preprocessing = choose_preprocessing(base_experiment, str(CONFIG.get("preprocessing", baseline_data.get("preprocessing", "raw"))))\n'
        '    requested_window = int(CONFIG.get("window_size", baseline_data.get("window_size", 128)))\n'
        '    target_window = choose_window(base_experiment, requested_window)\n'
        "    return {\n"
        '        "data": {\n'
            '            "preprocessing": target_preprocessing,\n'
            '            "window_size": target_window,\n'
        '            "notes": f\"{CONFIG.get(\'notes\', \'\')} | {strategy_summary(base_experiment)}\",\n'
        "        }\n"
        "    }\n"
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
                    "notes": str(config.get("notes", "")),
                }
            }
        elif agent_name == "augmentation_agent":
            patch = {
                "augmentation": {
                    "name": str(config["name"]),
                    "strength": str(config["strength"]),
                    "notes": str(config.get("notes", "")),
                }
            }
        else:
            patch = {
                "data": {
                    "preprocessing": str(config["preprocessing"]),
                    "window_size": int(config["window_size"]),
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
    updated["model"]["notes"] = proposal.notes
    return updated


def apply_augmentation_proposal(
    experiment: dict[str, object],
    proposal: AugmentationProposal,
) -> dict[str, object]:
    updated = clone_experiment(experiment)
    updated["augmentation"]["name"] = proposal.augmentation
    updated["augmentation"]["strength"] = proposal.strength
    updated["augmentation"]["notes"] = proposal.notes
    return updated


def apply_data_proposal(
    experiment: dict[str, object],
    proposal: DataProposal,
) -> dict[str, object]:
    updated = clone_experiment(experiment)
    updated["data"]["preprocessing"] = proposal.preprocessing
    updated["data"]["window_size"] = proposal.window_size
    updated["data"]["notes"] = proposal.notes
    return updated


def experiment_to_candidate_config(experiment: dict[str, object], notes: str) -> CandidateConfig:
    return CandidateConfig(
        model_name=str(experiment["model"]["name"]),
        preprocessing=str(experiment["data"]["preprocessing"]),
        augmentation=str(experiment["augmentation"]["name"]),
        augmentation_strength=str(experiment["augmentation"].get("strength", "medium")),
        learning_rate=float(experiment["model"]["learning_rate"]),
        window_size=int(experiment["data"]["window_size"]),
        notes=notes,
    )


def clone_experiment(experiment: dict[str, object]) -> dict[str, object]:
    return {
        "dataset": dict(experiment["dataset"]),
        "model": dict(experiment["model"]),
        "data": dict(experiment["data"]),
        "augmentation": dict(experiment["augmentation"]),
        "training": dict(experiment["training"]),
    }
