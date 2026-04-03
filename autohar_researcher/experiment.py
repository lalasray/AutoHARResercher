from __future__ import annotations

from autohar_researcher.types import (
    AugmentationProposal,
    CandidateConfig,
    DataProposal,
    ModelProposal,
)


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
            "epochs": 5,
            "metric": "macro_f1",
        },
    }


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
