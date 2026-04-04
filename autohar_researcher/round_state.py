from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class RoundState:
    round_id: int
    planner_artifact: str = ""
    sync_artifact: str = ""
    memory_artifacts: dict[str, str] = field(default_factory=dict)
    proposal_artifacts: dict[str, str] = field(default_factory=dict)
    composed_artifact: str = ""
    promotion_artifact: str = ""
    baseline_branch: str = ""
    synced_branches: list[str] = field(default_factory=list)
    materialized_agent_files: dict[str, str] = field(default_factory=dict)
    materialized_orchestrator_file: str = ""
    orchestrator_branch: str = ""
    orchestrator_worktree: str = ""
    combination_order: list[str] = field(default_factory=list)
    pushed_branches: list[str] = field(default_factory=list)
