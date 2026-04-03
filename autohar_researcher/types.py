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


@dataclass(frozen=True)
class AgentWorkspace:
    agent_name: str
    branch_name: str
    worktree_path: str


@dataclass(frozen=True)
class GitAction:
    label: str
    command: str
