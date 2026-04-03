from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from autohar_researcher.artifacts import ensure_round_dir, write_json_artifact
from autohar_researcher.git_runtime import GitRuntime
from autohar_researcher.harness import DummyHARHarness
from autohar_researcher.round_state import RoundState
from autohar_researcher.types import (
    AgentWorkspace,
    AugmentationProposal,
    CandidateConfig,
    CandidateResult,
    DataProposal,
    GitAction,
    ModelProposal,
)


class ModelAgent:
    name = "model_agent"

    def propose(self, round_id: int, best_score: float | None) -> ModelProposal:
        if best_score is None:
            return ModelProposal(
                model_name="cnn_gru",
                learning_rate=3e-4,
                notes="Balanced architecture-first guess for IMU HAR.",
            )
        return ModelProposal(
            model_name="cnn_gru",
            learning_rate=3e-4,
            notes="Push the strongest architecture toward a better-performing region.",
        )


class AugmentationAgent:
    name = "augmentation_agent"

    def propose(self, round_id: int, best_score: float | None) -> AugmentationProposal:
        if round_id % 2 == 1:
            return AugmentationProposal(
                augmentation="time_warp",
                notes="Probe a recurrent baseline with a stronger augmentation move.",
            )
        return AugmentationProposal(
            augmentation="rotation",
            notes="Test whether rotation augmentation helps a compact sequence model.",
        )


class DataAgent:
    name = "data_agent"

    def propose(self, round_id: int, best_score: float | None) -> DataProposal:
        if best_score is not None and best_score > 0.84:
            return DataProposal(
                preprocessing="normalized+gravity_filter",
                window_size=128,
                notes="Stabilize preprocessing and windowing around the best data setup.",
            )
        return DataProposal(
            preprocessing="normalized+magnitude_channels",
            window_size=256,
            notes="Probe an alternative data pipeline with richer derived channels.",
        )


class HierarchicalOrchestrator:
    def __init__(self, harness: DummyHARHarness) -> None:
        self.harness = harness
        self.agents = [ModelAgent(), AugmentationAgent(), DataAgent()]
        self.workspaces = self._build_workspaces()
        self.git_runtime = GitRuntime()
        self.orchestrator_workspace = AgentWorkspace(
            agent_name="orchestrator",
            branch_name="orchestrator-main",
            worktree_path=str(Path.cwd() / ".worktrees" / "orchestrator_main"),
        )

    def run(
        self,
        rounds: int = 3,
        materialize_git: bool = False,
        auto_push_non_main: bool = False,
    ) -> dict[str, object]:
        history: list[CandidateResult] = []
        best_result: CandidateResult | None = None
        actions: list[GitAction] = self.bootstrap_actions()
        round_states: list[RoundState] = []

        if materialize_git:
            self.git_runtime.bootstrap_agent_workspaces(self.workspaces)
            self.git_runtime.bootstrap_orchestrator_workspace(
                orchestrator_branch=self.orchestrator_workspace.branch_name,
                orchestrator_worktree=self.orchestrator_workspace.worktree_path,
            )

        for round_id in range(1, rounds + 1):
            round_plan = self.plan_round(round_id=round_id, best_result=best_result)
            actions.extend(self.plan_git_actions(round_id=round_id, round_plan=round_plan))
            round_results, round_state = self.execute_round(
                round_id=round_id,
                round_plan=round_plan,
                materialize_git=materialize_git,
                auto_push_non_main=auto_push_non_main,
            )
            history.extend(round_results)
            round_states.append(round_state)
            round_best = self.evaluate_round(round_results)
            if best_result is None or round_best.val_score > best_result.val_score:
                best_result = round_best

        assert best_result is not None
        actions.extend(self.finalize_actions(best_result=best_result))
        return {
            "best_result": best_result,
            "history": history,
            "workspaces": self.workspaces,
            "git_actions": actions,
            "round_states": round_states,
        }

    def _build_workspaces(self) -> dict[str, AgentWorkspace]:
        root = Path.cwd()
        worktree_root = root / ".worktrees"
        return {
            "model_agent": AgentWorkspace(
                agent_name="model_agent",
                branch_name="agent-model-main",
                worktree_path=str(worktree_root / "model_agent"),
            ),
            "augmentation_agent": AgentWorkspace(
                agent_name="augmentation_agent",
                branch_name="agent-augmentation-main",
                worktree_path=str(worktree_root / "augmentation_agent"),
            ),
            "data_agent": AgentWorkspace(
                agent_name="data_agent",
                branch_name="agent-data-main",
                worktree_path=str(worktree_root / "data_agent"),
            ),
        }

    def bootstrap_actions(self) -> list[GitAction]:
        actions: list[GitAction] = []
        for workspace in self.workspaces.values():
            actions.append(
                GitAction(
                    label=f"Create {workspace.agent_name} branch",
                    command=f"git branch {workspace.branch_name} main",
                )
            )
            actions.append(
                GitAction(
                    label=f"Create {workspace.agent_name} worktree",
                    command=f"git worktree add {workspace.worktree_path} {workspace.branch_name}",
                )
            )
        return actions

    def plan_round(
        self,
        round_id: int,
        best_result: CandidateResult | None,
    ) -> dict[str, object]:
        return {
            "round_id": round_id,
            "best_score": best_result.val_score if best_result else None,
            "baseline_branch": "main" if best_result is None else best_result.branch_name,
        }

    def plan_git_actions(
        self,
        round_id: int,
        round_plan: dict[str, object],
    ) -> list[GitAction]:
        actions: list[GitAction] = []
        baseline_branch = str(round_plan["baseline_branch"])
        for workspace in self.workspaces.values():
            actions.append(
                GitAction(
                    label=f"Round {round_id}: update {workspace.agent_name}",
                    command=(
                        f"git -C {workspace.worktree_path} commit -am "
                        f"\"round {round_id}: candidate from {workspace.agent_name}\""
                    ),
                )
            )
            actions.append(
                GitAction(
                    label=f"Round {round_id}: fetch proposal from {workspace.agent_name}",
                    command=(
                        f"git -C {workspace.worktree_path} diff {baseline_branch} "
                        f"# inspect {workspace.agent_name} contribution against {baseline_branch}"
                    ),
                )
            )
        actions.append(
            GitAction(
                    label=f"Round {round_id}: compose unified candidate",
                    command=(
                        f"git -C {self.orchestrator_workspace.worktree_path} commit -am "
                        f"\"round {round_id}: orchestrator candidate on top of {baseline_branch}\""
                    ),
                )
            )
        actions.append(
            GitAction(
                label=f"Round {round_id}: evaluate unified candidate",
                command="python3 -m autohar_researcher.main --rounds 1 # evaluate composed config",
            )
        )
        return actions

    def execute_round(
        self,
        round_id: int,
        round_plan: dict[str, object],
        materialize_git: bool = False,
        auto_push_non_main: bool = False,
    ) -> tuple[list[CandidateResult], RoundState]:
        best_score = round_plan["best_score"]
        with ThreadPoolExecutor(max_workers=len(self.agents)) as pool:
            futures = [
                pool.submit(agent.propose, round_id, best_score)
                for agent in self.agents
            ]
            proposals = [future.result() for future in futures]

        round_state = self._materialize_round_state(round_id=round_id, proposals=proposals)
        if materialize_git:
            round_state = self.git_runtime.materialize_round(
                round_id=round_id,
                baseline_branch=str(round_plan["baseline_branch"]),
                proposals=proposals,
                workspaces=self.workspaces,
                round_state=round_state,
                auto_push_non_main=auto_push_non_main,
            )
        result = self._compose_round_result(
            round_id=round_id,
            proposals=proposals,
            round_state=round_state,
        )
        return [result], round_state

    def evaluate_round(self, round_results: list[CandidateResult]) -> CandidateResult:
        return max(round_results, key=lambda result: result.val_score)

    def _compose_round_result(
        self,
        round_id: int,
        proposals: list[ModelProposal | AugmentationProposal | DataProposal],
        round_state: RoundState,
    ) -> CandidateResult:
        model_proposal = next(
            proposal for proposal in proposals if isinstance(proposal, ModelProposal)
        )
        augmentation_proposal = next(
            proposal for proposal in proposals if isinstance(proposal, AugmentationProposal)
        )
        data_proposal = next(
            proposal for proposal in proposals if isinstance(proposal, DataProposal)
        )

        config = CandidateConfig(
            model_name=model_proposal.model_name,
            preprocessing=data_proposal.preprocessing,
            augmentation=augmentation_proposal.augmentation,
            learning_rate=model_proposal.learning_rate,
            window_size=data_proposal.window_size,
            notes=(
                "Composed by orchestrator from model, augmentation, and data proposals."
            ),
        )
        component_summaries = {
            "model_agent": model_proposal.notes,
            "augmentation_agent": augmentation_proposal.notes,
            "data_agent": data_proposal.notes,
        }
        workspace = AgentWorkspace(
            agent_name="orchestrator",
            branch_name=round_state.orchestrator_branch or self.orchestrator_workspace.branch_name,
            worktree_path=round_state.orchestrator_worktree or self.orchestrator_workspace.worktree_path,
        )
        return self.harness.evaluate(
            round_id=round_id,
            agent_name="orchestrator",
            branch_name=workspace.branch_name,
            worktree_path=workspace.worktree_path,
            config=config,
            component_summaries=component_summaries,
            artifact_paths={
                **round_state.proposal_artifacts,
                "composed_candidate": round_state.composed_artifact,
                **{
                    f"materialized_{agent_name}": path
                    for agent_name, path in round_state.materialized_agent_files.items()
                },
                "materialized_orchestrator_candidate": round_state.materialized_orchestrator_file,
            },
        )

    def _materialize_round_state(
        self,
        round_id: int,
        proposals: list[ModelProposal | AugmentationProposal | DataProposal],
    ) -> RoundState:
        round_dir = ensure_round_dir(round_id)
        proposal_dir = round_dir / "agent_proposals"
        proposal_artifacts: dict[str, str] = {}

        for proposal in proposals:
            if isinstance(proposal, ModelProposal):
                agent_name = "model_agent"
            elif isinstance(proposal, AugmentationProposal):
                agent_name = "augmentation_agent"
            else:
                agent_name = "data_agent"
            proposal_artifacts[agent_name] = write_json_artifact(
                proposal_dir / f"{agent_name}.json",
                proposal,
            )

        model_proposal = next(
            proposal for proposal in proposals if isinstance(proposal, ModelProposal)
        )
        augmentation_proposal = next(
            proposal for proposal in proposals if isinstance(proposal, AugmentationProposal)
        )
        data_proposal = next(
            proposal for proposal in proposals if isinstance(proposal, DataProposal)
        )
        composed_payload = {
            "branch_name": self.orchestrator_workspace.branch_name,
            "source_branches": {
                "model_agent": self.workspaces["model_agent"].branch_name,
                "augmentation_agent": self.workspaces["augmentation_agent"].branch_name,
                "data_agent": self.workspaces["data_agent"].branch_name,
            },
            "composed_config": {
                "model_name": model_proposal.model_name,
                "learning_rate": model_proposal.learning_rate,
                "augmentation": augmentation_proposal.augmentation,
                "preprocessing": data_proposal.preprocessing,
                "window_size": data_proposal.window_size,
            },
        }
        composed_artifact = write_json_artifact(
            round_dir / "orchestrator_candidate.json",
            composed_payload,
        )
        return RoundState(
            round_id=round_id,
            proposal_artifacts=proposal_artifacts,
            composed_artifact=composed_artifact,
            orchestrator_branch=self.orchestrator_workspace.branch_name,
            orchestrator_worktree=self.orchestrator_workspace.worktree_path,
        )

    def finalize_actions(self, best_result: CandidateResult) -> list[GitAction]:
        return [
            GitAction(
                label="Promote best orchestrated branch",
                command=f"git merge --no-ff {best_result.branch_name}",
            ),
            GitAction(
                label="Rebase agent branches onto winning baseline",
                command=(
                    f"git rebase {best_result.branch_name} agent-model-main && "
                    f"git rebase {best_result.branch_name} agent-augmentation-main && "
                    f"git rebase {best_result.branch_name} agent-data-main"
                ),
            ),
        ]
