from __future__ import annotations

import os
from dataclasses import dataclass


DEFAULT_COLORS = {
    "model_agent": "#0b7285",
    "augmentation_agent": "#e67700",
    "data_agent": "#2b8a3e",
    "orchestrator": "#c2255c",
}


def load_reporting_colors() -> dict[str, str]:
    return {
        "model_agent": os.getenv(
            "AUTOHAR_COLOR_MODEL_AGENT", DEFAULT_COLORS["model_agent"]
        ),
        "augmentation_agent": os.getenv(
            "AUTOHAR_COLOR_AUGMENTATION_AGENT", DEFAULT_COLORS["augmentation_agent"]
        ),
        "data_agent": os.getenv(
            "AUTOHAR_COLOR_DATA_AGENT", DEFAULT_COLORS["data_agent"]
        ),
        "orchestrator": os.getenv(
            "AUTOHAR_COLOR_ORCHESTRATOR", DEFAULT_COLORS["orchestrator"]
        ),
    }


@dataclass(frozen=True)
class JudgeThresholds:
    prior_best_delta: float
    recent_window_delta: float
    relaxed_prior_best_delta: float
    relaxed_recent_delta: float


DEFAULT_JUDGE_THRESHOLDS = JudgeThresholds(
    prior_best_delta=0.015,
    recent_window_delta=0.01,
    relaxed_prior_best_delta=0.03,
    relaxed_recent_delta=0.02,
)


def load_judge_thresholds() -> JudgeThresholds:
    return JudgeThresholds(
        prior_best_delta=float(os.getenv("AUTOHAR_JUDGE_PRIOR_BEST_DELTA", "0.015")),
        recent_window_delta=float(os.getenv("AUTOHAR_JUDGE_RECENT_DELTA", "0.01")),
        relaxed_prior_best_delta=float(
            os.getenv("AUTOHAR_JUDGE_RELAXED_PRIOR_BEST_DELTA", "0.03")
        ),
        relaxed_recent_delta=float(
            os.getenv("AUTOHAR_JUDGE_RELAXED_RECENT_DELTA", "0.02")
        ),
    )


@dataclass(frozen=True)
class TrainingConfig:
    base_lr_multiplier: float
    mlp_lr_multiplier: float
    grad_clip_mlp: float
    grad_clip_linear: float
    lr_decay_factor: float
    lr_decay_patience: int


DEFAULT_TRAINING_CONFIG = TrainingConfig(
    base_lr_multiplier=50.0,
    mlp_lr_multiplier=50.0,
    grad_clip_mlp=5.0,
    grad_clip_linear=2.0,
    lr_decay_factor=0.9,
    lr_decay_patience=10,
)


def load_training_config() -> TrainingConfig:
    return TrainingConfig(
        base_lr_multiplier=float(os.getenv("AUTOHAR_BASE_LR_MULTIPLIER", "50.0")),
        mlp_lr_multiplier=float(os.getenv("AUTOHAR_MLP_LR_MULTIPLIER", "50.0")),
        grad_clip_mlp=float(os.getenv("AUTOHAR_GRAD_CLIP_MLP", "5.0")),
        grad_clip_linear=float(os.getenv("AUTOHAR_GRAD_CLIP_LINEAR", "2.0")),
        lr_decay_factor=float(os.getenv("AUTOHAR_LR_DECAY_FACTOR", "0.9")),
        lr_decay_patience=int(os.getenv("AUTOHAR_LR_DECAY_PATIENCE", "10")),
    )


@dataclass(frozen=True)
class SandboxConfig:
    timeout_seconds: int
    memory_mb: int
    mode: str


DEFAULT_SANDBOX_CONFIG = SandboxConfig(
    timeout_seconds=5,
    memory_mb=128,
    mode="thread",
)


def load_sandbox_config() -> SandboxConfig:
    return SandboxConfig(
        timeout_seconds=int(os.getenv("AUTOHAR_SANDBOX_TIMEOUT", "5")),
        memory_mb=int(os.getenv("AUTOHAR_SANDBOX_MEMORY_MB", "128")),
        mode=os.getenv("AUTOHAR_SANDBOX_MODE", "thread"),
    )
