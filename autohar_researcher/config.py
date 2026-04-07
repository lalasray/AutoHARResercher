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


@dataclass(frozen=True)
class RepairConfig:
    max_retries: int
    backoff_base: float
    fallback_to_heuristic: bool
    max_patch_retries: int


DEFAULT_REPAIR_CONFIG = RepairConfig(
    max_retries=3,
    backoff_base=2.0,
    fallback_to_heuristic=True,
    max_patch_retries=3,
)


def load_repair_config() -> RepairConfig:
    return RepairConfig(
        max_retries=int(os.getenv("AUTOHAR_REPAIR_MAX_RETRIES", "3")),
        backoff_base=float(os.getenv("AUTOHAR_REPAIR_BACKOFF_BASE", "2.0")),
        fallback_to_heuristic=os.getenv(
            "AUTOHAR_REPAIR_FALLBACK_HEURISTIC", "true"
        ).lower()
        == "true",
        max_patch_retries=int(os.getenv("AUTOHAR_REPAIR_MAX_PATCH_RETRIES", "3")),
    )


@dataclass(frozen=True)
class TimeBudgetConfig:
    enabled: bool
    budget_seconds: float
    startup_overhead_seconds: float


DEFAULT_TIME_BUDGET_CONFIG = TimeBudgetConfig(
    enabled=False,
    budget_seconds=300.0,
    startup_overhead_seconds=10.0,
)


def load_time_budget_config() -> TimeBudgetConfig:
    enabled = os.getenv("AUTOHAR_TIME_BUDGET_ENABLED", "false").lower() == "true"
    if not enabled:
        return DEFAULT_TIME_BUDGET_CONFIG
    return TimeBudgetConfig(
        enabled=True,
        budget_seconds=float(os.getenv("AUTOHAR_TIME_BUDGET_SECONDS", "300")),
        startup_overhead_seconds=float(
            os.getenv("AUTOHAR_TIME_BUDGET_STARTUP_OVERHEAD", "10")
        ),
    )


@dataclass(frozen=True)
class AutoresearchConfig:
    never_stop: bool
    simplification_threshold: float
    experiment_branch_prefix: str
    auto_restart_on_crash: bool


DEFAULT_AUTORESEARCH_CONFIG = AutoresearchConfig(
    never_stop=False,
    simplification_threshold=0.001,
    experiment_branch_prefix="autoresearch",
    auto_restart_on_crash=True,
)


def load_autoresearch_config() -> AutoresearchConfig:
    never_stop = os.getenv("AUTOHAR_NEVER_STOP", "false").lower() == "true"
    return AutoresearchConfig(
        never_stop=never_stop,
        simplification_threshold=float(
            os.getenv("AUTOHAR_SIMPLIFICATION_THRESHOLD", "0.001")
        ),
        experiment_branch_prefix=os.getenv(
            "AUTOHAR_EXPERIMENT_BRANCH_PREFIX", "autoresearch"
        ),
        auto_restart_on_crash=os.getenv("AUTOHAR_AUTO_RESTART_ON_CRASH", "true").lower()
        == "true",
    )
