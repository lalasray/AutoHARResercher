from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class CandidateConfig:
    model_name: str
    preprocessing: str
    augmentation: str
    learning_rate: float
    window_size: int
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
    component_summaries: dict[str, str] = field(default_factory=dict)
    artifact_paths: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class ModelProposal:
    model_name: str
    learning_rate: float
    notes: str = ""


@dataclass(frozen=True)
class AugmentationProposal:
    augmentation: str
    notes: str = ""


@dataclass(frozen=True)
class DataProposal:
    preprocessing: str
    window_size: int
    notes: str = ""


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
    history_csv: str
    history_svg: str
