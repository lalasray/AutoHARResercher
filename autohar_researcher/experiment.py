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
    body = pprint.pformat(payload, sort_dicts=True, width=88)
    path.write_text(
        '"""Agent-owned module config for AutoHARResearcher."""\n\n'
        f"MODULE_PAYLOAD = {body}\n",
        encoding="utf-8",
    )


def load_module_python(path: Path) -> dict[str, object]:
    namespace: dict[str, object] = {}
    exec(path.read_text(encoding="utf-8"), {}, namespace)
    payload = namespace.get("MODULE_PAYLOAD", {})
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid module payload in {path}")
    return payload


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
        payload = load_module_python(module_path)
        config = payload.get("config", {})
        if agent_name == "model_agent":
            experiment["model"]["name"] = str(config.get("name", experiment["model"]["name"]))
            experiment["model"]["learning_rate"] = float(
                config.get("learning_rate", experiment["model"]["learning_rate"])
            )
            if "notes" in config:
                experiment["model"]["notes"] = str(config.get("notes", ""))
        elif agent_name == "augmentation_agent":
            experiment["augmentation"]["name"] = str(
                config.get("name", experiment["augmentation"]["name"])
            )
            experiment["augmentation"]["strength"] = str(
                config.get("strength", experiment["augmentation"]["strength"])
            )
            if "notes" in config:
                experiment["augmentation"]["notes"] = str(config.get("notes", ""))
        else:
            experiment["data"]["preprocessing"] = str(
                config.get("preprocessing", experiment["data"]["preprocessing"])
            )
            experiment["data"]["window_size"] = int(
                config.get("window_size", experiment["data"]["window_size"])
            )
            if "notes" in config:
                experiment["data"]["notes"] = str(config.get("notes", ""))
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
