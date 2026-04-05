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
    compose_experiment_from_workspace,
    load_module_python,
    module_owned_fields,
    module_payload,
    module_relative_path,
    write_module_python,
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
        sync_report: dict[str, object] = {
            "round_id": round_id,
            "baseline_branch": baseline_branch,
            "synced_workspaces": {},
        }
        baseline_experiment = self._load_baseline_experiment(
            baseline_branch=baseline_branch,
            orchestrator_worktree=Path(round_state.orchestrator_worktree),
        )
        baseline_module_payloads = self._load_baseline_module_payloads(
            baseline_branch=baseline_branch,
            orchestrator_worktree=Path(round_state.orchestrator_worktree),
            workspaces=workspaces,
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
            experiment_rel = Path("research") / "experiment.json"
            module_rel = module_relative_path(agent_name)
            sync_payload = {
                "round_id": round_id,
                "agent_name": agent_name,
                "branch_name": workspace.branch_name,
                "sync_from_branch": baseline_branch,
                "owned_fields": module_owned_fields(agent_name),
                "experiment": clone_experiment(baseline_experiment),
            }
            experiment_abs = Path(workspace.worktree_path) / experiment_rel
            self.write_json(experiment_abs, sync_payload)
            workspace_sync_paths: dict[str, str] = {
                "experiment": str(experiment_abs),
            }
            for sync_agent_name in workspaces:
                sync_module_abs = Path(workspace.worktree_path) / module_relative_path(sync_agent_name)
                write_module_python(
                    sync_module_abs,
                    module_payload(
                        sync_agent_name,
                        baseline_experiment,
                        round_id=round_id,
                        branch_name=workspace.branch_name,
                        sync_from_branch=baseline_branch,
                        logic_expression=str(
                            baseline_module_payloads.get(sync_agent_name, {}).get("logic_expression", "")
                        ),
                        helper_code=str(
                            baseline_module_payloads.get(sync_agent_name, {}).get("helper_code", "")
                        ),
                        function_rewrite=str(
                            baseline_module_payloads.get(sync_agent_name, {}).get("function_rewrite", "")
                        ),
                    ),
                )
                workspace_sync_paths[sync_agent_name] = str(sync_module_abs)
            sync_committed = self.commit_paths_if_changed(
                worktree=Path(workspace.worktree_path),
                rel_paths=[experiment_rel, *[module_relative_path(name) for name in workspaces]],
                message=f"round {round_id}: sync from {baseline_branch}",
            )
            sync_report["synced_workspaces"][agent_name] = {
                "branch_name": workspace.branch_name,
                "worktree_path": workspace.worktree_path,
                "paths": workspace_sync_paths,
                "committed_sync": sync_committed,
            }
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
                "owned_fields": module_owned_fields(agent_name),
                "config": module_payload(
                    agent_name,
                    experiment,
                    round_id=round_id,
                    branch_name=workspace.branch_name,
                    sync_from_branch=baseline_branch,
                    logic_expression=str(proposal.logic_expression),
                    helper_code=str(proposal.helper_code),
                    function_rewrite=str(proposal.function_rewrite),
                )["config"],
                "logic_expression": str(proposal.logic_expression),
                "helper_code": str(proposal.helper_code),
                "function_rewrite": str(proposal.function_rewrite),
            }
            module_abs = Path(workspace.worktree_path) / module_rel
            write_module_python(module_abs, payload)
            committed = self.commit_paths_if_changed(
                worktree=Path(workspace.worktree_path),
                rel_paths=[module_rel],
                message=f"round {round_id}: {agent_name} proposal",
            )
            round_state.materialized_agent_files[agent_name] = str(module_abs)
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
            source_payload[agent_name] = load_module_python(Path(materialized_path))

        composed_experiment = clone_experiment(baseline_experiment)
        composed_experiment["model"] = dict(agent_experiments["model_agent"]["model"])
        composed_experiment["data"] = dict(agent_experiments["data_agent"]["data"])
        composed_experiment["augmentation"] = dict(agent_experiments["augmentation_agent"]["augmentation"])

        orchestrator_rel = Path("research") / "experiment.json"
        orchestrator_module_rels = [module_relative_path(name) for name in workspaces]
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
        for source_agent_name in workspaces:
            write_module_python(
                orchestrator_worktree / module_relative_path(source_agent_name),
                module_payload(
                    source_agent_name,
                    composed_experiment,
                    round_id=round_id,
                    branch_name=orchestrator_branch,
                    sync_from_branch=baseline_branch,
                    logic_expression=str(
                        source_payload.get(source_agent_name, {}).get("logic_expression", "")
                    ),
                    helper_code=str(
                        source_payload.get(source_agent_name, {}).get("helper_code", "")
                    ),
                    function_rewrite=str(
                        source_payload.get(source_agent_name, {}).get("function_rewrite", "")
                    ),
                ),
            )
        committed = self.commit_paths_if_changed(
            worktree=orchestrator_worktree,
            rel_paths=[orchestrator_rel, *orchestrator_module_rels],
            message=f"round {round_id}: composed orchestrator candidate",
        )
        round_state.materialized_orchestrator_file = str(orchestrator_abs)
        if auto_push_non_main and committed and orchestrator_branch != "main":
            self.push_branch(orchestrator_branch, cwd=orchestrator_worktree)
            round_state.pushed_branches.append(orchestrator_branch)
        round_dir = self.repo_root / "artifacts" / f"round_{round_id}"
        round_dir.mkdir(parents=True, exist_ok=True)
        sync_artifact = round_dir / "baseline_sync.json"
        self.write_json(sync_artifact, sync_report)
        round_state.sync_artifact = str(sync_artifact)
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
        return self.commit_paths_if_changed(worktree=worktree, rel_paths=[rel_path], message=message)

    def commit_paths_if_changed(self, worktree: Path, rel_paths: list[Path], message: str) -> bool:
        status = self.run_git(
            ["status", "--short", "--", *[str(rel_path) for rel_path in rel_paths]],
            cwd=worktree,
        )
        if not status.stdout.strip():
            return False
        self.run_git(["add", *[str(rel_path) for rel_path in rel_paths]], cwd=worktree)
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

    def _load_baseline_experiment(
        self,
        baseline_branch: str,
        orchestrator_worktree: Path,
    ) -> dict[str, object]:
        if baseline_branch == "main":
            return base_experiment_config()
        if baseline_branch == "orchestrator-main":
            return compose_experiment_from_workspace(
                orchestrator_worktree,
                fallback_experiment=base_experiment_config(),
                apply_modules=False,
            )
        else:
            branch_directory = baseline_branch.replace("agent-", "").replace("-main", "")
            if branch_directory == "augmentation":
                branch_directory = "augmentation_agent"
            elif branch_directory == "data":
                branch_directory = "data_agent"
            elif branch_directory == "model":
                branch_directory = "model_agent"
            branch_worktree = self.repo_root / ".worktrees" / branch_directory
            owned_agents: list[str] | None = None
            if baseline_branch == "agent-model-main":
                owned_agents = ["model_agent"]
            elif baseline_branch == "agent-augmentation-main":
                owned_agents = ["augmentation_agent"]
            elif baseline_branch == "agent-data-main":
                owned_agents = ["data_agent"]
            return compose_experiment_from_workspace(
                branch_worktree,
                fallback_experiment=base_experiment_config(),
                apply_modules=owned_agents is not None,
                module_agents=owned_agents,
            )

    def _load_baseline_module_payloads(
        self,
        baseline_branch: str,
        orchestrator_worktree: Path,
        workspaces: dict[str, AgentWorkspace],
    ) -> dict[str, dict[str, object]]:
        if baseline_branch == "main":
            return {}
        if baseline_branch == "orchestrator-main":
            baseline_worktree = orchestrator_worktree
        else:
            branch_directory = baseline_branch.replace("agent-", "").replace("-main", "")
            if branch_directory == "augmentation":
                branch_directory = "augmentation_agent"
            elif branch_directory == "data":
                branch_directory = "data_agent"
            elif branch_directory == "model":
                branch_directory = "model_agent"
            baseline_worktree = self.repo_root / ".worktrees" / branch_directory

        payloads: dict[str, dict[str, object]] = {}
        for agent_name in workspaces:
            module_path = baseline_worktree / module_relative_path(agent_name)
            if module_path.exists():
                payloads[agent_name] = load_module_python(module_path)
        return payloads

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
