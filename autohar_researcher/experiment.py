from __future__ import annotations

import json
from pathlib import Path

from autohar_researcher.types import (
    AugmentationProposal,
    CandidateConfig,
    DataProposal,
    ModelProposal,
)
from autohar_researcher.experiment_catalog import (
    AVAILABLE_AUGMENTATIONS,
    AVAILABLE_PREPROCESSING,
    MODEL_ARCHITECTURES,
    MODEL_HYPERPARAMETERS,
    available_augmentations,
    available_model_architectures,
    available_preprocessing_modes,
    default_augmentation_params,
    default_data_params,
    default_model_params,
    model_architecture_catalog,
)
from autohar_researcher.experiment_templates import write_module_python

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
            "epochs": 30,
            "min_epochs": 4,
            "early_stopping_patience": 5,
            "device": "auto",
            "metric": "macro_f1",
        },
    }


def module_relative_path(agent_name: str) -> Path:
    return Path("research") / "modules" / MODULE_FILENAMES[agent_name]


def module_owned_fields(agent_name: str) -> list[str]:
    return {
        "model_agent": ["model.name", "model.learning_rate", "model.params"],
        "augmentation_agent": [
            "augmentation.name",
            "augmentation.strength",
            "augmentation.params",
        ],
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
    *,
    apply_modules: bool = True,
    module_agents: list[str] | tuple[str, ...] | None = None,
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

    if not apply_modules:
        return experiment

    agent_names = (
        list(module_agents) if module_agents is not None else list(MODULE_FILENAMES)
    )
    for agent_name in agent_names:
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
    updated["model"]["params"] = (
        dict(proposal.model_params)
        if proposal.model_params
        else default_model_params(proposal.model_name)
    )
    if proposal.model_name == "custom" and proposal.function_rewrite:
        updated["model"]["params"]["custom_code"] = proposal.function_rewrite
    updated["model"]["notes"] = proposal.notes
    return updated


def apply_augmentation_proposal(
    experiment: dict[str, object],
    proposal: AugmentationProposal,
) -> dict[str, object]:
    updated = clone_experiment(experiment)
    updated["augmentation"]["name"] = proposal.augmentation
    updated["augmentation"]["strength"] = proposal.strength
    params = default_augmentation_params(
        proposal.augmentation,
        proposal.strength,
    )
    if proposal.augmentation == "custom" and proposal.function_rewrite:
        params["custom_code"] = proposal.function_rewrite
    updated["augmentation"]["params"] = params
    updated["augmentation"]["notes"] = proposal.notes
    return updated


def apply_data_proposal(
    experiment: dict[str, object],
    proposal: DataProposal,
) -> dict[str, object]:
    updated = clone_experiment(experiment)
    updated["data"]["preprocessing"] = proposal.preprocessing
    updated["data"]["window_size"] = proposal.window_size
    params = default_data_params(
        proposal.preprocessing,
        proposal.window_size,
    )
    params.update(dict(proposal.data_params))
    if proposal.preprocessing == "custom" and proposal.function_rewrite:
        params["custom_code"] = proposal.function_rewrite
    updated["data"]["params"] = params
    updated["data"]["notes"] = proposal.notes
    return updated


def experiment_to_candidate_config(
    experiment: dict[str, object], notes: str
) -> CandidateConfig:
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
