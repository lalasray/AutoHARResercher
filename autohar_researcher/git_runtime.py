from __future__ import annotations

import json
import subprocess
from dataclasses import asdict, is_dataclass
from pathlib import Path

from autohar_researcher.round_state import RoundState
from autohar_researcher.types import (
    AgentWorkspace,
    AugmentationProposal,
    DataProposal,
    ModelProposal,
)


class GitRuntime:
    def __init__(self, repo_root: Path | None = None) -> None:
        self.repo_root = repo_root or Path.cwd()

    def bootstrap_agent_workspaces(self, workspaces: dict[str, AgentWorkspace]) -> None:
        for workspace in workspaces.values():
            self.ensure_branch(workspace.branch_name, start_point="main")
            self.ensure_worktree(workspace.worktree_path, workspace.branch_name)

    def materialize_round(
        self,
        round_id: int,
        baseline_branch: str,
        proposals: list[ModelProposal | AugmentationProposal | DataProposal],
        workspaces: dict[str, AgentWorkspace],
        round_state: RoundState,
    ) -> RoundState:
        self.bootstrap_agent_workspaces(workspaces)
        round_state.baseline_branch = baseline_branch

        for proposal in proposals:
            if isinstance(proposal, ModelProposal):
                agent_name = "model_agent"
            elif isinstance(proposal, AugmentationProposal):
                agent_name = "augmentation_agent"
            else:
                agent_name = "data_agent"

            workspace = workspaces[agent_name]
            rel_path = Path("research") / "proposals" / f"round_{round_id}.json"
            payload = {
                "round_id": round_id,
                "agent_name": agent_name,
                "branch_name": workspace.branch_name,
                "proposal": self._to_serializable(proposal),
            }
            abs_path = Path(workspace.worktree_path) / rel_path
            self.write_json(abs_path, payload)
            self.commit_if_changed(
                worktree=Path(workspace.worktree_path),
                rel_path=rel_path,
                message=f"round {round_id}: {agent_name} proposal",
            )
            round_state.materialized_agent_files[agent_name] = str(abs_path)

        orchestrator_branch = f"orchestrator-round-{round_id}"
        orchestrator_worktree = self.repo_root / ".worktrees" / f"orchestrator_round_{round_id}"
        self.ensure_branch(orchestrator_branch, start_point=baseline_branch)
        self.ensure_worktree(str(orchestrator_worktree), orchestrator_branch)

        source_payload = {}
        for agent_name, materialized_path in round_state.materialized_agent_files.items():
            source_payload[agent_name] = json.loads(Path(materialized_path).read_text(encoding="utf-8"))

        orchestrator_rel = Path("research") / "composed" / f"round_{round_id}.json"
        composed_payload = {
            "round_id": round_id,
            "branch_name": orchestrator_branch,
            "baseline_branch": baseline_branch,
            "source_branches": {
                name: workspace.branch_name for name, workspace in workspaces.items()
            },
            "source_files": round_state.materialized_agent_files,
            "composed_from": source_payload,
        }
        orchestrator_abs = orchestrator_worktree / orchestrator_rel
        self.write_json(orchestrator_abs, composed_payload)
        self.commit_if_changed(
            worktree=orchestrator_worktree,
            rel_path=orchestrator_rel,
            message=f"round {round_id}: composed orchestrator candidate",
        )
        round_state.orchestrator_branch = orchestrator_branch
        round_state.orchestrator_worktree = str(orchestrator_worktree)
        round_state.materialized_orchestrator_file = str(orchestrator_abs)
        return round_state

    def ensure_branch(self, branch_name: str, start_point: str) -> None:
        exists = self.run_git(["rev-parse", "--verify", branch_name], check=False)
        if exists.returncode != 0:
            self.run_git(["branch", branch_name, start_point])

    def ensure_worktree(self, worktree_path: str, branch_name: str) -> None:
        target = str(Path(worktree_path).resolve())
        listed = self.run_git(["worktree", "list", "--porcelain"])
        if f"worktree {target}" in listed.stdout:
            return
        self.run_git(["worktree", "add", worktree_path, branch_name])

    def commit_if_changed(self, worktree: Path, rel_path: Path, message: str) -> None:
        status = self.run_git(
            ["status", "--short", "--", str(rel_path)],
            cwd=worktree,
        )
        if not status.stdout.strip():
            return
        self.run_git(["add", str(rel_path)], cwd=worktree)
        self.run_git(["commit", "-m", message], cwd=worktree)

    def write_json(self, path: Path, payload: object) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self._to_serializable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")

    def _to_serializable(self, payload: object) -> object:
        if is_dataclass(payload):
            return asdict(payload)
        return payload

    def run_git(
        self,
        args: list[str],
        cwd: Path | None = None,
        check: bool = True,
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["git", *args],
            cwd=cwd or self.repo_root,
            check=check,
            text=True,
            capture_output=True,
        )
