from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from autohar_researcher.harness import DummyHARHarness
from autohar_researcher.types import AgentWorkspace, CandidateConfig, CandidateResult, GitAction


class ModelAgent:
    name = "model_agent"

    def propose(self, round_id: int, best_score: float | None) -> CandidateConfig:
        if best_score is None:
            return CandidateConfig(
                model_name="cnn_gru",
                preprocessing="normalized",
                augmentation="jitter",
                learning_rate=3e-4,
                window_size=128,
                notes="Balanced architecture-first guess for IMU HAR.",
            )
        return CandidateConfig(
            model_name="cnn_gru",
            preprocessing="normalized+gravity_filter",
            augmentation="rotation",
            learning_rate=3e-4,
            window_size=128,
            notes="Push the strongest architecture toward a better-performing region.",
        )


class AugmentationAgent:
    name = "augmentation_agent"

    def propose(self, round_id: int, best_score: float | None) -> CandidateConfig:
        if round_id % 2 == 1:
            return CandidateConfig(
                model_name="gru",
                preprocessing="normalized+magnitude_channels",
                augmentation="time_warp",
                learning_rate=1e-3,
                window_size=128,
                notes="Probe a recurrent baseline with a stronger augmentation move.",
            )
        return CandidateConfig(
            model_name="tiny_transformer",
            preprocessing="normalized",
            augmentation="rotation",
            learning_rate=3e-4,
            window_size=64,
            notes="Test whether rotation augmentation helps a compact sequence model.",
        )


class DataAgent:
    name = "data_agent"

    def propose(self, round_id: int, best_score: float | None) -> CandidateConfig:
        if best_score is not None and best_score > 0.84:
            return CandidateConfig(
                model_name="cnn_gru",
                preprocessing="normalized+gravity_filter",
                augmentation="none",
                learning_rate=1e-4,
                window_size=128,
                notes="Stabilize preprocessing and windowing around the best data setup.",
            )
        return CandidateConfig(
            model_name="cnn",
            preprocessing="normalized+magnitude_channels",
            augmentation="mixup",
            learning_rate=3e-4,
            window_size=256,
            notes="Probe an alternative data pipeline with richer derived channels.",
        )


class HierarchicalOrchestrator:
    def __init__(self, harness: DummyHARHarness) -> None:
        self.harness = harness
        self.agents = [ModelAgent(), AugmentationAgent(), DataAgent()]
        self.workspaces = self._build_workspaces()

    def run(self, rounds: int = 3) -> dict[str, object]:
        history: list[CandidateResult] = []
        best_result: CandidateResult | None = None
        actions: list[GitAction] = self.bootstrap_actions()

        for round_id in range(1, rounds + 1):
            round_plan = self.plan_round(round_id=round_id, best_result=best_result)
            actions.extend(self.plan_git_actions(round_id=round_id, round_plan=round_plan))
            round_results = self.execute_round(round_id=round_id, round_plan=round_plan)
            history.extend(round_results)
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
        }

    def _build_workspaces(self) -> dict[str, AgentWorkspace]:
        root = Path.cwd()
        worktree_root = root / ".worktrees"
        return {
            "model_agent": AgentWorkspace(
                agent_name="model_agent",
                branch_name="agent/model/main",
                worktree_path=str(worktree_root / "model_agent"),
            ),
            "augmentation_agent": AgentWorkspace(
                agent_name="augmentation_agent",
                branch_name="agent/augmentation/main",
                worktree_path=str(worktree_root / "augmentation_agent"),
            ),
            "data_agent": AgentWorkspace(
                agent_name="data_agent",
                branch_name="agent/data/main",
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
    ) -> dict[str, float | int | None]:
        return {
            "round_id": round_id,
            "best_score": best_result.val_score if best_result else None,
        }

    def plan_git_actions(
        self,
        round_id: int,
        round_plan: dict[str, float | int | None],
    ) -> list[GitAction]:
        actions: list[GitAction] = []
        best_score = round_plan["best_score"]
        baseline_note = "main" if best_score is None else "best-so-far branch"
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
                    label=f"Round {round_id}: evaluate {workspace.agent_name}",
                    command=(
                        f"python3 -m autohar_researcher.main --rounds 1 "
                        f"# from {workspace.branch_name} using {baseline_note}"
                    ),
                )
            )
        return actions

    def execute_round(
        self,
        round_id: int,
        round_plan: dict[str, float | int | None],
    ) -> list[CandidateResult]:
        best_score = round_plan["best_score"]
        with ThreadPoolExecutor(max_workers=len(self.agents)) as pool:
            futures = [
                pool.submit(self._run_agent_round, agent, round_id, best_score)
                for agent in self.agents
            ]
            return [future.result() for future in futures]

    def evaluate_round(self, round_results: list[CandidateResult]) -> CandidateResult:
        return max(round_results, key=lambda result: result.val_score)

    def _run_agent_round(
        self,
        agent: ModelAgent | AugmentationAgent | DataAgent,
        round_id: int,
        best_score: float | None,
    ) -> CandidateResult:
        config = agent.propose(round_id=round_id, best_score=best_score)
        workspace = self.workspaces[agent.name]
        return self.harness.evaluate(
            round_id=round_id,
            agent_name=agent.name,
            branch_name=workspace.branch_name,
            worktree_path=workspace.worktree_path,
            config=config,
        )

    def finalize_actions(self, best_result: CandidateResult) -> list[GitAction]:
        return [
            GitAction(
                label="Promote best agent branch",
                command=f"git merge --no-ff {best_result.branch_name}",
            )
        ]
