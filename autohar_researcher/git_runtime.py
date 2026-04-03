from __future__ import annotations

import json
import subprocess
from dataclasses import asdict, is_dataclass
from pathlib import Path

from autohar_researcher.experiment import (
    apply_augmentation_proposal,
    apply_data_proposal,
    apply_model_proposal,
    base_experiment_config,
    clone_experiment,
)
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

    def bootstrap_orchestrator_workspace(
        self,
        orchestrator_branch: str,
        orchestrator_worktree: str,
    ) -> None:
        self.ensure_branch(orchestrator_branch, start_point="main")
        self.ensure_worktree(orchestrator_worktree, orchestrator_branch)

    def materialize_round(
        self,
        round_id: int,
        baseline_branch: str,
        proposals: list[ModelProposal | AugmentationProposal | DataProposal],
        workspaces: dict[str, AgentWorkspace],
        round_state: RoundState,
        auto_push_non_main: bool = False,
    ) -> RoundState:
        self.bootstrap_agent_workspaces(workspaces)
        round_state.baseline_branch = baseline_branch
        baseline_experiment = self._load_baseline_experiment(
            baseline_branch=baseline_branch,
            orchestrator_worktree=Path(round_state.orchestrator_worktree),
        )
        agent_experiments: dict[str, dict[str, object]] = {}

        for proposal in proposals:
            if isinstance(proposal, ModelProposal):
                agent_name = "model_agent"
                experiment = apply_model_proposal(clone_experiment(baseline_experiment), proposal)
            elif isinstance(proposal, AugmentationProposal):
                agent_name = "augmentation_agent"
                experiment = apply_augmentation_proposal(clone_experiment(baseline_experiment), proposal)
            else:
                agent_name = "data_agent"
                experiment = apply_data_proposal(clone_experiment(baseline_experiment), proposal)

            workspace = workspaces[agent_name]
            rel_path = Path("research") / "experiment.json"
            sync_payload = {
                "round_id": round_id,
                "agent_name": agent_name,
                "branch_name": workspace.branch_name,
                "sync_from_branch": baseline_branch,
                "owned_fields": self._owned_fields(agent_name),
                "experiment": clone_experiment(baseline_experiment),
            }
            abs_path = Path(workspace.worktree_path) / rel_path
            self.write_json(abs_path, sync_payload)
            sync_committed = self.commit_if_changed(
                worktree=Path(workspace.worktree_path),
                rel_path=rel_path,
                message=f"round {round_id}: sync from {baseline_branch}",
            )
            if sync_committed:
                round_state.synced_branches.append(workspace.branch_name)
                if auto_push_non_main:
                    self.push_branch(workspace.branch_name, cwd=Path(workspace.worktree_path))
                    round_state.pushed_branches.append(workspace.branch_name)

            payload = {
                "round_id": round_id,
                "agent_name": agent_name,
                "branch_name": workspace.branch_name,
                "sync_from_branch": baseline_branch,
                "owned_fields": self._owned_fields(agent_name),
                "experiment": experiment,
            }
            self.write_json(abs_path, payload)
            committed = self.commit_if_changed(
                worktree=Path(workspace.worktree_path),
                rel_path=rel_path,
                message=f"round {round_id}: {agent_name} proposal",
            )
            round_state.materialized_agent_files[agent_name] = str(abs_path)
            agent_experiments[agent_name] = experiment
            if auto_push_non_main and committed:
                self.push_branch(workspace.branch_name, cwd=Path(workspace.worktree_path))
                round_state.pushed_branches.append(workspace.branch_name)

        orchestrator_branch = round_state.orchestrator_branch
        orchestrator_worktree = Path(round_state.orchestrator_worktree)
        self.ensure_branch(orchestrator_branch, start_point="main")
        self.ensure_worktree(str(orchestrator_worktree), orchestrator_branch)

        source_payload = {}
        for agent_name, materialized_path in round_state.materialized_agent_files.items():
            source_payload[agent_name] = json.loads(Path(materialized_path).read_text(encoding="utf-8"))

        composed_experiment = clone_experiment(baseline_experiment)
        composed_experiment["model"] = dict(agent_experiments["model_agent"]["model"])
        composed_experiment["data"] = dict(agent_experiments["data_agent"]["data"])
        composed_experiment["augmentation"] = dict(agent_experiments["augmentation_agent"]["augmentation"])

        orchestrator_rel = Path("research") / "experiment.json"
        composed_payload = {
            "round_id": round_id,
            "branch_name": orchestrator_branch,
            "baseline_branch": baseline_branch,
            "source_branches": {
                name: workspace.branch_name for name, workspace in workspaces.items()
            },
            "source_files": round_state.materialized_agent_files,
            "composed_from": source_payload,
            "experiment": composed_experiment,
        }
        orchestrator_abs = orchestrator_worktree / orchestrator_rel
        self.write_json(orchestrator_abs, composed_payload)
        committed = self.commit_if_changed(
            worktree=orchestrator_worktree,
            rel_path=orchestrator_rel,
            message=f"round {round_id}: composed orchestrator candidate",
        )
        round_state.materialized_orchestrator_file = str(orchestrator_abs)
        if auto_push_non_main and committed and orchestrator_branch != "main":
            self.push_branch(orchestrator_branch, cwd=orchestrator_worktree)
            round_state.pushed_branches.append(orchestrator_branch)
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

    def commit_if_changed(self, worktree: Path, rel_path: Path, message: str) -> bool:
        status = self.run_git(
            ["status", "--short", "--", str(rel_path)],
            cwd=worktree,
        )
        if not status.stdout.strip():
            return False
        self.run_git(["add", str(rel_path)], cwd=worktree)
        self.run_git(["commit", "-m", message], cwd=worktree)
        return True

    def push_branch(self, branch_name: str, cwd: Path | None = None) -> None:
        if branch_name == "main" or not self.has_remote("origin"):
            return
        self.run_git(["push", "-u", "origin", branch_name], cwd=cwd)

    def write_json(self, path: Path, payload: object) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self._to_serializable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")

    def _to_serializable(self, payload: object) -> object:
        if is_dataclass(payload):
            return asdict(payload)
        return payload

    def has_remote(self, remote_name: str) -> bool:
        remote = self.run_git(["remote", "get-url", remote_name], check=False)
        return remote.returncode == 0 and bool(remote.stdout.strip())

    def _owned_fields(self, agent_name: str) -> list[str]:
        owned = {
            "model_agent": ["model.name", "model.learning_rate"],
            "augmentation_agent": ["augmentation.name"],
            "data_agent": ["data.preprocessing", "data.window_size"],
        }
        return owned[agent_name]

    def _load_baseline_experiment(
        self,
        baseline_branch: str,
        orchestrator_worktree: Path,
    ) -> dict[str, object]:
        if baseline_branch == "main":
            return base_experiment_config()
        experiment_path = orchestrator_worktree / "research" / "experiment.json"
        if not experiment_path.exists():
            return base_experiment_config()
        payload = json.loads(experiment_path.read_text(encoding="utf-8"))
        if "experiment" in payload:
            return payload["experiment"]
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
