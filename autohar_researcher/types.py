from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class CandidateConfig:
    model_name: str
    preprocessing: str
    augmentation: str
    augmentation_strength: str
    learning_rate: float
    window_size: int
    augmentation_params: dict[str, float | int | str] = field(default_factory=dict)
    notes: str = ""


@dataclass
class CandidateResult:
    round_id: int
    agent_name: str
    branch_name: str
    worktree_path: str
    config: CandidateConfig
    val_score: float
    summary: str
    score_breakdown: dict[str, float] = field(default_factory=dict)
    details: dict[str, object] = field(default_factory=dict)
    component_summaries: dict[str, str] = field(default_factory=dict)
    artifact_paths: dict[str, str] = field(default_factory=dict)
    accepted: bool = False
    best_so_far_score: float = 0.0


@dataclass(frozen=True)
class ModelProposal:
    model_name: str
    learning_rate: float
    notes: str = ""
    logic_expression: str = ""
    helper_code: str = ""
    function_rewrite: str = ""


@dataclass(frozen=True)
class AugmentationProposal:
    augmentation: str
    strength: str
    notes: str = ""
    logic_expression: str = ""
    helper_code: str = ""
    function_rewrite: str = ""


@dataclass(frozen=True)
class DataProposal:
    preprocessing: str
    window_size: int
    notes: str = ""
    logic_expression: str = ""
    helper_code: str = ""
    function_rewrite: str = ""


@dataclass(frozen=True)
class AgentWorkspace:
    agent_name: str
    branch_name: str
    worktree_path: str


@dataclass(frozen=True)
class GitAction:
    label: str
    command: str


@dataclass(frozen=True)
class RunOutputs:
    trial_history_csv: str
    trial_history_svg: str
    accepted_history_csv: str
    accepted_history_svg: str
    latest_promotion_artifact: str = ""


@dataclass(frozen=True)
class ModelPlanHint:
    model_name: str
    learning_rate: float
    rationale: str = ""
    logic_expression: str = ""
    helper_code: str = ""
    function_rewrite: str = ""


@dataclass(frozen=True)
class AugmentationPlanHint:
    augmentation: str
    strength: str
    rationale: str = ""
    logic_expression: str = ""
    helper_code: str = ""
    function_rewrite: str = ""


@dataclass(frozen=True)
class DataPlanHint:
    preprocessing: str
    window_size: int
    rationale: str = ""
    logic_expression: str = ""
    helper_code: str = ""
    function_rewrite: str = ""
