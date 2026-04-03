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
            "def build_patch(base_experiment: dict[str, object]) -> dict[str, object]:\n"
            '    """Return the model-only patch owned by ModelAgent."""\n'
            "    del base_experiment\n"
            "    return {\n"
            '        "model": {\n'
            '            "name": str(CONFIG["name"]),\n'
            '            "learning_rate": float(CONFIG["learning_rate"]),\n'
            '            "notes": str(CONFIG.get("notes", "")),\n'
            "        }\n"
            "    }\n"
        )
    if agent_name == "augmentation_agent":
        return (
            "def build_patch(base_experiment: dict[str, object]) -> dict[str, object]:\n"
            '    """Return the augmentation-only patch owned by AugmentationAgent."""\n'
            "    del base_experiment\n"
            "    return {\n"
            '        "augmentation": {\n'
            '            "name": str(CONFIG["name"]),\n'
            '            "strength": str(CONFIG["strength"]),\n'
            '            "notes": str(CONFIG.get("notes", "")),\n'
            "        }\n"
            "    }\n"
        )
    return (
        "def build_patch(base_experiment: dict[str, object]) -> dict[str, object]:\n"
        '    """Return the data-only patch owned by DataAgent."""\n'
        "    del base_experiment\n"
        "    return {\n"
        '        "data": {\n'
        '            "preprocessing": str(CONFIG["preprocessing"]),\n'
        '            "window_size": int(CONFIG["window_size"]),\n'
        '            "notes": str(CONFIG.get("notes", "")),\n'
        "        }\n"
        "    }\n"
    )


def load_module_python(path: Path) -> dict[str, object]:
    namespace: dict[str, object] = {}
    exec(path.read_text(encoding="utf-8"), {}, namespace)
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
    exec(path.read_text(encoding="utf-8"), {}, namespace)
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
