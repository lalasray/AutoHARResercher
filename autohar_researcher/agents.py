from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path

from autohar_researcher.artifacts import ensure_round_dir, write_json_artifact
from autohar_researcher.experiment import (
    apply_augmentation_proposal,
    apply_data_proposal,
    apply_model_proposal,
    base_experiment_config,
    clone_experiment,
    compose_experiment_from_workspace,
    default_model_params,
    experiment_to_candidate_config,
)
from autohar_researcher.git_runtime import GitRuntime
from autohar_researcher.harness import DummyHARHarness
from autohar_researcher.planner import build_planner
from autohar_researcher.reporting import (
    write_history_csv,
    write_history_svg,
    write_promotion_svg,
)
from autohar_researcher.round_state import RoundState
from autohar_researcher.types import (
    AgentWorkspace,
    AugmentationProposal,
    CandidateConfig,
    CandidateResult,
    DataProposal,
    DataPlanHint,
    GitAction,
    ModelPlanHint,
    ModelProposal,
    RunOutputs,
    AugmentationPlanHint,
)


def _dedupe_preserve_order(items: list[object]) -> list[object]:
    ordered: list[object] = []
    for item in items:
        if item not in ordered:
            ordered.append(item)
    return ordered


class ModelAgent:
    name = "model_agent"
    MODEL_CANDIDATES = [
        "deepconvlstm",
        "deepconvlstmattention",
        "hart",
        "tinierhar",
        "tinyhar",
        "spectra",
        "cnn_gru",
        "cnn",
        "tiny_transformer",
        "gru",
    ]

    def propose(
        self,
        round_id: int,
        best_score: float | None,
        best_result: CandidateResult | None,
        history: list[CandidateResult],
        accepted_history: list[CandidateResult],
        planner_hint: ModelPlanHint | None = None,
    ) -> ModelProposal:
        del best_score
        if planner_hint is not None:
            return ModelProposal(
                model_name=planner_hint.model_name,
                learning_rate=planner_hint.learning_rate,
                model_params=dict(planner_hint.model_params),
                notes=f"LLM planner hint: {planner_hint.rationale}",
                logic_expression=planner_hint.logic_expression,
                helper_code=planner_hint.helper_code,
                function_rewrite=planner_hint.function_rewrite,
            )
        tried = {
            (result.config.model_name, round(result.config.learning_rate, 7))
            for result in history
            if result.agent_name == self.name
        }
        best_config = best_result.config if best_result else None
        base_model = best_config.model_name if best_config else "deepconvlstm"
        base_lr = best_config.learning_rate if best_config else 3e-4
        learning_rates = [
            max(1e-4, round(base_lr * factor, 6))
            for factor in (0.5, 0.8, 1.0, 1.25, 1.5)
        ]
        candidate_order = [
            (base_model, learning_rate) for learning_rate in learning_rates
        ]
        candidate_order.extend(
            (model_name, learning_rate)
            for model_name in [base_model, *self.MODEL_CANDIDATES]
            for learning_rate in learning_rates[:3]
        )
        candidate_order = _dedupe_preserve_order(candidate_order)
        model_name, learning_rate = self._pick_candidate(
            candidate_order=candidate_order,
            tried=tried,
            round_id=round_id,
        )
        note_prefix = "Mutate around accepted best model." if accepted_history else "Seed the first strong architecture."
        return ModelProposal(
            model_name=model_name,
            learning_rate=learning_rate,
            model_params=default_model_params(model_name),
            notes=f"{note_prefix} Exploring {model_name} at lr={learning_rate}.",
            logic_expression="",
            helper_code="",
            function_rewrite="",
        )

    def _pick_candidate(
        self,
        candidate_order: list[tuple[str, float]],
        tried: set[tuple[str, float]],
        round_id: int,
    ) -> tuple[str, float]:
        for candidate in candidate_order:
            if candidate not in tried:
                return candidate
        return candidate_order[(round_id - 1) % len(candidate_order)]


class AugmentationAgent:
    name = "augmentation_agent"
    AUGMENTATIONS = [
        "none",
        "rotation",
        "time_warp",
        "jitter",
        "mixup",
        "scaling",
        "drift",
        "permutation",
        "magnitude_warp",
    ]
    STRENGTHS = ["weak", "medium", "strong"]

    def propose(
        self,
        round_id: int,
        best_score: float | None,
        best_result: CandidateResult | None,
        history: list[CandidateResult],
        accepted_history: list[CandidateResult],
        planner_hint: AugmentationPlanHint | None = None,
    ) -> AugmentationProposal:
        del best_score
        if planner_hint is not None:
            return AugmentationProposal(
                augmentation=planner_hint.augmentation,
                strength=planner_hint.strength,
                notes=f"LLM planner hint: {planner_hint.rationale}",
                logic_expression=planner_hint.logic_expression,
                helper_code=planner_hint.helper_code,
                function_rewrite=planner_hint.function_rewrite,
            )
        best_config = best_result.config if best_result else None
        accepted_aug = best_config.augmentation if best_config else "none"
        accepted_strength = best_config.augmentation_strength if best_config else "medium"
        context_key = self._context_key(best_config)
        tried = {
            (result.config.augmentation, result.config.augmentation_strength)
            for result in history
            if result.agent_name == self.name
            and self._context_key(result.config) == context_key
        }
        candidates = [
            (augmentation, strength)
            for augmentation in self.AUGMENTATIONS
            for strength in self.STRENGTHS
        ]
        context_ranked = sorted(
            candidates,
            key=lambda candidate: (
                self._contextual_score(history, accepted_history, candidate[0], candidate[1], context_key),
                self._global_score(history, accepted_history, candidate[0], candidate[1]),
            ),
            reverse=True,
        )
        global_ranked = sorted(
            candidates,
            key=lambda candidate: self._global_score(history, accepted_history, candidate[0], candidate[1]),
            reverse=True,
        )
        ranked = [(accepted_aug, accepted_strength)]
        if accepted_aug == "none":
            ranked.extend(candidate for candidate in context_ranked if candidate[0] != "none")
            ranked.extend([("none", strength) for strength in self.STRENGTHS if ("none", strength) != (accepted_aug, accepted_strength)])
        else:
            ranked.extend(context_ranked)
        ranked.extend(global_ranked)
        ranked.extend(candidates)
        ranked = _dedupe_preserve_order(ranked)
        augmentation, strength = self._pick_option(ranked, tried, round_id)
        note_prefix = (
            "Use context-aware augmentation ranking around the accepted baseline."
            if accepted_history
            else "Sweep the first augmentation choices."
        )
        return AugmentationProposal(
            augmentation=augmentation,
            strength=strength,
            notes=(
                f"{note_prefix} Testing augmentation={augmentation} "
                f"strength={strength} for context={context_key}."
            ),
            logic_expression="",
            helper_code="",
            function_rewrite="",
        )

    def _context_key(self, config: CandidateConfig | None) -> str:
        if config is None:
            return "bootstrap"
        return f"{config.model_name}|{config.preprocessing}|{int(config.window_size)}"

    def _contextual_score(
        self,
        history: list[CandidateResult],
        accepted_history: list[CandidateResult],
        augmentation: str,
        strength: str,
        context_key: str,
    ) -> float:
        scores = [
            result.val_score
            for result in [*history, *accepted_history]
            if self._context_key(result.config) == context_key
            and result.config.augmentation == augmentation
            and result.config.augmentation_strength == strength
            and result.agent_name in {self.name, "orchestrator"}
        ]
        return max(scores, default=float("-inf"))

    def _global_score(
        self,
        history: list[CandidateResult],
        accepted_history: list[CandidateResult],
        augmentation: str,
        strength: str,
    ) -> float:
        scores = [
            result.val_score
            for result in history
            if result.config.augmentation == augmentation
            and result.config.augmentation_strength == strength
            and result.agent_name in {self.name, "orchestrator"}
        ]
        scores.extend(
            result.val_score
            for result in accepted_history
            if result.config.augmentation == augmentation
            and result.config.augmentation_strength == strength
        )
        return max(scores, default=float("-inf"))

    def _pick_option(
        self,
        ranked: list[tuple[str, str]],
        tried: set[tuple[str, str]],
        round_id: int,
    ) -> tuple[str, str]:
        for option in ranked:
            if option not in tried:
                return option
        return ranked[(round_id - 1) % len(ranked)]


class DataAgent:
    name = "data_agent"
    PREPROCESSING_CANDIDATES = [
        "raw",
        "normalized",
        "normalized+gravity_filter",
        "normalized+magnitude_channels",
    ]
    WINDOW_CANDIDATES = [64, 96, 128, 192, 256]

    def propose(
        self,
        round_id: int,
        best_score: float | None,
        best_result: CandidateResult | None,
        history: list[CandidateResult],
        accepted_history: list[CandidateResult],
        planner_hint: DataPlanHint | None = None,
    ) -> DataProposal:
        del best_score
        if planner_hint is not None:
            return DataProposal(
                preprocessing=planner_hint.preprocessing,
                window_size=planner_hint.window_size,
                notes=f"LLM planner hint: {planner_hint.rationale}",
                logic_expression=planner_hint.logic_expression,
                helper_code=planner_hint.helper_code,
                function_rewrite=planner_hint.function_rewrite,
            )
        tried = {
            (result.config.preprocessing, int(result.config.window_size))
            for result in history
            if result.agent_name == self.name
        }
        best_config = best_result.config if best_result else None
        base_preprocessing = best_config.preprocessing if best_config else "raw"
        base_window = best_config.window_size if best_config else 128
        candidate_order: list[tuple[str, int]] = [
            (base_preprocessing, window_size)
            for window_size in self._rank_windows(base_window)
        ]
        candidate_order.extend(
            (preprocessing, base_window)
            for preprocessing in [base_preprocessing, *self.PREPROCESSING_CANDIDATES]
        )
        scored_options = sorted(
            (
                (
                    self._best_score_for_option(history, preprocessing, window_size),
                    preprocessing,
                    window_size,
                )
                for preprocessing in self.PREPROCESSING_CANDIDATES
                for window_size in self.WINDOW_CANDIDATES
            ),
            reverse=True,
        )
        candidate_order.extend((preprocessing, window_size) for _, preprocessing, window_size in scored_options)
        candidate_order = _dedupe_preserve_order(candidate_order)
        preprocessing, window_size = self._pick_candidate(candidate_order, tried, round_id)
        note_prefix = "Mutate around accepted preprocessing/window choices." if accepted_history else "Probe initial preprocessing and window choices."
        return DataProposal(
            preprocessing=preprocessing,
            window_size=window_size,
            notes=f"{note_prefix} Testing {preprocessing} with window={window_size}.",
            logic_expression="",
            helper_code="",
            function_rewrite="",
        )

    def _rank_windows(self, base_window: int) -> list[int]:
        return sorted(self.WINDOW_CANDIDATES, key=lambda candidate: (abs(candidate - base_window), candidate))

    def _best_score_for_option(
        self,
        history: list[CandidateResult],
        preprocessing: str,
        window_size: int,
    ) -> float:
        scores = [
            result.val_score
            for result in history
            if result.agent_name == self.name
            and result.config.preprocessing == preprocessing
            and int(result.config.window_size) == window_size
        ]
        return max(scores, default=float("-inf"))

    def _pick_candidate(
        self,
        candidate_order: list[tuple[str, int]],
        tried: set[tuple[str, int]],
        round_id: int,
    ) -> tuple[str, int]:
        for candidate in candidate_order:
            if candidate not in tried:
                return candidate
        return candidate_order[(round_id - 1) % len(candidate_order)]


class HierarchicalOrchestrator:
    def __init__(self, harness: DummyHARHarness) -> None:
        self.harness = harness
        self.agents = [ModelAgent(), AugmentationAgent(), DataAgent()]
        self.planner = build_planner()
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
        resume: bool = False,
    ) -> dict[str, object]:
        if resume:
            history, accepted_history = self._load_round_history()
            if accepted_history:
                best_result = self._clone_result(accepted_history[-1])
            elif history:
                best_result = self._clone_result(max(history, key=lambda item: item.val_score))
            else:
                best_result = None
            best_by_agent = self._best_by_agent_from_history(history)
            start_round = (max((result.round_id for result in history), default=0) + 1)
        else:
            history = []
            accepted_history = []
            best_result = None
            best_by_agent = {}
            start_round = 1
        actions: list[GitAction] = self.bootstrap_actions()
        round_states: list[RoundState] = []
        run_outputs = self._write_run_outputs(history, accepted_history)

        if materialize_git:
            self.git_runtime.bootstrap_agent_workspaces(self.workspaces)
            self.git_runtime.bootstrap_orchestrator_workspace(
                orchestrator_branch=self.orchestrator_workspace.branch_name,
                orchestrator_worktree=self.orchestrator_workspace.worktree_path,
            )

        final_round = start_round + rounds - 1
        for round_id in range(start_round, final_round + 1):
            print(f"[AutoHAR] round {round_id}/{final_round}", flush=True)
            previous_best = None if best_result is None else self._clone_result(best_result)
            round_plan = self.plan_round(
                round_id=round_id,
                best_result=best_result,
                history=history,
                accepted_history=accepted_history,
            )
            actions.extend(self.plan_git_actions(round_id=round_id, round_plan=round_plan))
            round_results, round_state = self.execute_round(
                round_id=round_id,
                round_plan=round_plan,
                best_result=best_result,
                history=history,
                accepted_history=accepted_history,
                materialize_git=materialize_git,
                auto_push_non_main=auto_push_non_main,
            )
            self._annotate_best_so_far(round_results, best_by_agent)
            round_best = self.evaluate_round(round_results)
            accepted_candidate = self._select_accepted_candidate(round_results, best_result)
            if accepted_candidate is not None:
                accepted_candidate.accepted = True
                best_result = accepted_candidate
                accepted_history.append(self._clone_result(accepted_candidate))

            history.extend(round_results)
            round_state.promotion_artifact = self._write_promotion_artifact(
                round_id=round_id,
                round_results=round_results,
                round_best=round_best,
                previous_best=previous_best,
                accepted_candidate=accepted_candidate,
                current_best=best_result,
            )
            run_outputs = self._write_run_outputs(history, accepted_history)
            round_states.append(round_state)
            if best_result is None:
                best_result = round_best

        if best_result is None:
            raise RuntimeError(
                "No best result is available. Run at least one round, or use --resume only when completed round artifacts exist."
            )
        actions.extend(self.finalize_actions(best_result=best_result))
        return {
            "best_result": best_result,
            "history": history,
            "accepted_history": accepted_history,
            "workspaces": self.workspaces,
            "git_actions": actions,
            "round_states": round_states,
            "run_outputs": run_outputs,
            "planner_name": getattr(self.planner, "name", "unknown"),
        }

    def _load_round_history(self) -> tuple[list[CandidateResult], list[CandidateResult]]:
        history: list[CandidateResult] = []
        accepted_history: list[CandidateResult] = []
        artifacts_root = Path.cwd() / "artifacts"
        round_dirs = sorted(
            [
                path
                for path in artifacts_root.glob("round_*")
                if path.is_dir() and path.name.split("_")[-1].isdigit()
            ],
            key=lambda path: int(path.name.split("_")[-1]),
        )
        accepted_best = float("-inf")
        best_by_agent: dict[str, float] = {}

        for round_dir in round_dirs:
            round_id = int(round_dir.name.split("_")[-1])
            promotion_path = round_dir / "promotion.json"
            results_dir = round_dir / "results"
            if not promotion_path.exists() or not results_dir.exists():
                continue

            round_results: list[CandidateResult] = []
            for result_path in sorted(results_dir.glob("*.json")):
                payload = json.loads(result_path.read_text(encoding="utf-8"))
                config_payload = dict(payload["config"])
                score_breakdown_payload = {
                    key: float(value) for key, value in payload.get("score_breakdown", {}).items()
                }
                val_score = float(
                    score_breakdown_payload.get(
                        "macro_f1",
                        payload.get("val_score", 0.0),
                    )
                )
                result = CandidateResult(
                    round_id=round_id,
                    agent_name=str(payload["agent_name"]),
                    branch_name=str(payload["branch_name"]),
                    worktree_path=self._workspace_for_branch(str(payload["branch_name"])).worktree_path,
                    config=CandidateConfig(
                        model_name=str(config_payload["model_name"]),
                        preprocessing=str(config_payload["preprocessing"]),
                        augmentation=str(config_payload["augmentation"]),
                        augmentation_strength=str(config_payload["augmentation_strength"]),
                        learning_rate=float(config_payload["learning_rate"]),
                        window_size=int(config_payload["window_size"]),
                        model_params=dict(config_payload.get("model_params", {})),
                        data_params=dict(config_payload.get("data_params", {})),
                        augmentation_params=dict(config_payload.get("augmentation_params", {})),
                        notes=str(config_payload.get("notes", "")),
                    ),
                    val_score=val_score,
                    summary=str(payload.get("summary", "")),
                    score_breakdown=score_breakdown_payload,
                    details=dict(payload.get("details", {})),
                )
                best_by_agent[result.agent_name] = max(
                    best_by_agent.get(result.agent_name, float("-inf")),
                    result.val_score,
                )
                result.best_so_far_score = best_by_agent[result.agent_name]
                round_results.append(result)
                history.append(result)

            if not round_results:
                continue

            promotion_payload = json.loads(promotion_path.read_text(encoding="utf-8"))
            if promotion_payload.get("accepted_new_baseline"):
                winner = dict(promotion_payload.get("round_winner", {}))
                accepted_match = next(
                    (
                        result
                        for result in round_results
                        if result.agent_name == str(winner.get("agent_name", ""))
                        and result.branch_name == str(winner.get("branch_name", ""))
                    ),
                    None,
                )
                if accepted_match is not None:
                    accepted_match.accepted = True
                    accepted_best = max(accepted_best, accepted_match.val_score)
                    accepted_match.best_so_far_score = accepted_best
                    accepted_history.append(self._clone_result(accepted_match))

        return history, accepted_history

    def _best_by_agent_from_history(self, history: list[CandidateResult]) -> dict[str, float]:
        best_by_agent: dict[str, float] = {}
        for result in history:
            best_by_agent[result.agent_name] = max(
                best_by_agent.get(result.agent_name, float("-inf")),
                result.val_score,
            )
        return best_by_agent

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
        history: list[CandidateResult],
        accepted_history: list[CandidateResult],
    ) -> dict[str, object]:
        planner_output = self.planner.plan_round(
            round_id=round_id,
            best_result=best_result,
            history=history,
            accepted_history=accepted_history,
        )
        return {
            "round_id": round_id,
            "best_score": best_result.val_score if best_result else None,
            "baseline_branch": "main" if best_result is None else best_result.branch_name,
            "planner_name": planner_output.get("planner", getattr(self.planner, "name", "unknown")),
            "planner_summary": planner_output.get("summary", ""),
            "planner_hints": planner_output.get("hints", {}),
            "orchestrator_plan": planner_output.get("orchestrator_plan", {}),
            "agent_memories": planner_output.get("agent_memories", {}),
            "combination_memory": planner_output.get("combination_memory", {}),
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
        best_result: CandidateResult | None,
        history: list[CandidateResult],
        accepted_history: list[CandidateResult],
        materialize_git: bool = False,
        auto_push_non_main: bool = False,
    ) -> tuple[list[CandidateResult], RoundState]:
        best_score = round_plan["best_score"]
        planner_hints = dict(round_plan.get("planner_hints", {}))
        with ThreadPoolExecutor(max_workers=len(self.agents)) as pool:
            futures = [
                pool.submit(
                    agent.propose,
                    round_id,
                    best_score,
                    best_result,
                    list(history),
                    list(accepted_history),
                    planner_hints.get(agent.name),
                )
                for agent in self.agents
            ]
            proposals = [future.result() for future in futures]

        round_state = self._materialize_round_state(
            round_id=round_id,
            proposals=proposals,
            round_plan=round_plan,
        )
        if materialize_git:
            round_state = self.git_runtime.materialize_round(
                round_id=round_id,
                baseline_branch=str(round_plan["baseline_branch"]),
                proposals=proposals,
                workspaces=self.workspaces,
                round_state=round_state,
                auto_push_non_main=auto_push_non_main,
            )
        agent_results = self._evaluate_agent_results(
            round_id=round_id,
            proposals=proposals,
            round_state=round_state,
            baseline_branch=str(round_plan["baseline_branch"]),
        )
        result = self._evaluate_orchestrator_result(
            round_id=round_id,
            proposals=proposals,
            round_state=round_state,
            baseline_branch=str(round_plan["baseline_branch"]),
            round_plan=round_plan,
            history=history,
            agent_results=agent_results,
        )
        return [*agent_results, result], round_state

    def evaluate_round(self, round_results: list[CandidateResult]) -> CandidateResult:
        return max(round_results, key=lambda result: result.val_score)

    def _evaluate_orchestrator_result(
        self,
        round_id: int,
        proposals: list[ModelProposal | AugmentationProposal | DataProposal],
        round_state: RoundState,
        baseline_branch: str,
        round_plan: dict[str, object],
        history: list[CandidateResult],
        agent_results: list[CandidateResult],
    ) -> CandidateResult:
        baseline_experiment = self._baseline_experiment_from_branch(baseline_branch)
        eligible_agents = self._eligible_combination_agents(
            agent_results=agent_results,
            history=history,
        )
        candidate_specs = [
            ("model+augmentation", {"model_agent", "augmentation_agent"}),
            ("model+data", {"model_agent", "data_agent"}),
            ("augmentation+data", {"augmentation_agent", "data_agent"}),
            ("model+augmentation+data", {"model_agent", "augmentation_agent", "data_agent"}),
        ]
        candidate_specs = [
            (label, included_agents)
            for label, included_agents in candidate_specs
            if included_agents.issubset(eligible_agents)
        ]
        round_state.combination_order = [label for label, _ in candidate_specs]
        if not candidate_specs:
            return self._skipped_orchestrator_result(
                round_id=round_id,
                round_state=round_state,
                baseline_experiment=baseline_experiment,
                eligible_agents=eligible_agents,
            )
        candidate_specs = self._rank_candidate_specs(
            candidate_specs=candidate_specs,
            combination_memory=dict(round_plan.get("combination_memory", {})),
            orchestrator_plan=dict(round_plan.get("orchestrator_plan", {})),
        )
        round_state.combination_order = [label for label, _ in candidate_specs]
        evaluated: list[CandidateResult] = []
        for label, included_agents in candidate_specs:
            experiment = self._compose_experiment_from_proposals(
                proposals=proposals,
                baseline_experiment=baseline_experiment,
                included_agents=included_agents,
            )
            if (
                included_agents == {"model_agent", "augmentation_agent", "data_agent"}
                and round_state.materialized_orchestrator_file
            ):
                composed_payload = self._load_json(round_state.materialized_orchestrator_file)
                experiment = composed_payload["experiment"]
            config = experiment_to_candidate_config(
                experiment,
                notes=f"Orchestrator evaluated combination: {label}.",
            )
            component_summaries = self._component_summaries(proposals, included_agents)
            workspace = AgentWorkspace(
                agent_name="orchestrator",
                branch_name=round_state.orchestrator_branch or self.orchestrator_workspace.branch_name,
                worktree_path=round_state.orchestrator_worktree or self.orchestrator_workspace.worktree_path,
            )
            evaluated.append(
                self.harness.evaluate(
                    round_id=round_id,
                    agent_name="orchestrator",
                    branch_name=workspace.branch_name,
                    worktree_path=workspace.worktree_path,
                    config=config,
                    experiment=experiment,
                    component_summaries=component_summaries,
                    artifact_paths={
                        **round_state.proposal_artifacts,
                        "selected_combination": label,
                        "combination_order": ",".join(round_state.combination_order),
                        "composed_candidate": round_state.composed_artifact,
                        **{
                            f"materialized_{agent_name}": path
                            for agent_name, path in round_state.materialized_agent_files.items()
                        },
                        "materialized_orchestrator_candidate": round_state.materialized_orchestrator_file,
                    },
                )
            )
        best_candidate = max(evaluated, key=lambda result: result.val_score)
        best_candidate.summary = (
            f"{best_candidate.summary} | selected_combination="
            f"{best_candidate.artifact_paths.get('selected_combination', 'unknown')}"
        )
        best_candidate.artifact_paths["evaluation_report"] = self._write_result_artifact(best_candidate)
        return best_candidate

    def _eligible_combination_agents(
        self,
        agent_results: list[CandidateResult],
        history: list[CandidateResult],
    ) -> set[str]:
        prior_best_by_agent: dict[str, float] = {}
        for result in history:
            if result.agent_name not in {"model_agent", "augmentation_agent", "data_agent"}:
                continue
            prior_best_by_agent[result.agent_name] = max(
                prior_best_by_agent.get(result.agent_name, float("-inf")),
                result.val_score,
            )
        eligible: set[str] = set()
        for result in agent_results:
            if result.agent_name not in {"model_agent", "augmentation_agent", "data_agent"}:
                continue
            prior_best = prior_best_by_agent.get(result.agent_name, float("-inf"))
            if result.val_score > prior_best:
                eligible.add(result.agent_name)
        return eligible

    def _skipped_orchestrator_result(
        self,
        round_id: int,
        round_state: RoundState,
        baseline_experiment: dict[str, object],
        eligible_agents: set[str],
    ) -> CandidateResult:
        config = experiment_to_candidate_config(
            baseline_experiment,
            notes="Orchestrator skipped combination evaluation because no multi-agent improving set qualified.",
        )
        summary = (
            "orchestrator skipped combination evaluation because fewer than two specialist agents "
            f"beat their own historical bests; eligible_agents={sorted(eligible_agents)}"
        )
        result = CandidateResult(
            round_id=round_id,
            agent_name="orchestrator",
            branch_name=self.orchestrator_workspace.branch_name,
            worktree_path=self.orchestrator_workspace.worktree_path,
            config=config,
            val_score=0.0,
            summary=summary,
            artifact_paths={
                "selected_combination": "",
                "combination_order": ",".join(round_state.combination_order),
                "skip_reason": "no_eligible_multi_agent_combo",
                "eligible_agents": ",".join(sorted(eligible_agents)),
                "composed_candidate": round_state.composed_artifact,
            },
        )
        result.artifact_paths["evaluation_report"] = self._write_result_artifact(result)
        return result

    def _rank_candidate_specs(
        self,
        candidate_specs: list[tuple[str, set[str]]],
        combination_memory: dict[str, object],
        orchestrator_plan: dict[str, object],
    ) -> list[tuple[str, set[str]]]:
        accepted_best = dict(combination_memory.get("accepted_best_by_combination", {}))
        recent_best = dict(combination_memory.get("recent_best_by_combination", {}))
        recent_avg = dict(combination_memory.get("recent_avg_by_combination", {}))
        compatible = {
            str(item.get("label", "")): float(item.get("synergy", 0.0))
            for item in combination_memory.get("compatible_patterns", [])
            if isinstance(item, dict)
        }
        conflicting = {
            str(item.get("label", "")): float(item.get("synergy", 0.0))
            for item in combination_memory.get("conflicting_patterns", [])
            if isinstance(item, dict)
        }
        combination_bias = str(orchestrator_plan.get("combination_bias", "")).lower()

        def label_score(label: str) -> float:
            score = 0.0
            score += 3.0 * float(accepted_best.get(label, 0.0))
            score += 2.0 * float(recent_best.get(label, 0.0))
            score += 1.0 * float(recent_avg.get(label, 0.0))
            score += 2.0 * float(compatible.get(label, 0.0))
            score += 2.0 * float(conflicting.get(label, 0.0))
            is_pair = label.count("+") == 1
            is_three_way = label.count("+") == 2
            if combination_bias in {"pairwise", "stability"}:
                if is_pair:
                    score += 0.25
                if is_three_way:
                    score -= 0.25
            elif combination_bias in {"broader", "broad", "composition"}:
                if is_three_way:
                    score += 0.25
            return score

        return sorted(
            candidate_specs,
            key=lambda item: (label_score(item[0]), -item[0].count("+")),
            reverse=True,
        )

    def _evaluate_agent_results(
        self,
        round_id: int,
        proposals: list[ModelProposal | AugmentationProposal | DataProposal],
        round_state: RoundState,
        baseline_branch: str,
    ) -> list[CandidateResult]:
        baseline_experiment = self._baseline_experiment_from_branch(baseline_branch)
        results: list[CandidateResult] = []
        for proposal in proposals:
            if isinstance(proposal, ModelProposal):
                agent_name = "model_agent"
                experiment = apply_model_proposal(clone_experiment(baseline_experiment), proposal)
                notes = proposal.notes
            elif isinstance(proposal, AugmentationProposal):
                agent_name = "augmentation_agent"
                experiment = apply_augmentation_proposal(clone_experiment(baseline_experiment), proposal)
                notes = proposal.notes
            else:
                agent_name = "data_agent"
                experiment = apply_data_proposal(clone_experiment(baseline_experiment), proposal)
                notes = proposal.notes

            config = experiment_to_candidate_config(experiment, notes=notes)
            workspace = self.workspaces[agent_name]
            results.append(
                self.harness.evaluate(
                    round_id=round_id,
                    agent_name=agent_name,
                    branch_name=workspace.branch_name,
                    worktree_path=workspace.worktree_path,
                    config=config,
                    experiment=experiment,
                    artifact_paths={
                        "proposal_artifact": round_state.proposal_artifacts.get(agent_name, ""),
                        "materialized_agent_module": round_state.materialized_agent_files.get(agent_name, ""),
                    },
                )
            )
        return results

    def _materialize_round_state(
        self,
        round_id: int,
        proposals: list[ModelProposal | AugmentationProposal | DataProposal],
        round_plan: dict[str, object],
    ) -> RoundState:
        round_dir = ensure_round_dir(round_id)
        proposal_dir = round_dir / "agent_proposals"
        proposal_artifacts: dict[str, str] = {}
        planner_hints = round_plan.get("planner_hints", {})
        agent_memories = round_plan.get("agent_memories", {})
        combination_memory = round_plan.get("combination_memory", {})
        planner_payload = {
            "round_id": round_id,
            "planner_name": round_plan.get("planner_name", ""),
            "planner_summary": round_plan.get("planner_summary", ""),
            "baseline_branch": round_plan.get("baseline_branch", ""),
            "best_score": round_plan.get("best_score"),
            "orchestrator_plan": round_plan.get("orchestrator_plan", {}),
            "combination_memory": combination_memory,
            "planned_combination_order": [],
            "planner_hints": {
                "model_agent": (
                    {
                        "model_name": planner_hints["model_agent"].model_name,
                        "learning_rate": planner_hints["model_agent"].learning_rate,
                        "rationale": planner_hints["model_agent"].rationale,
                        "logic_expression": planner_hints["model_agent"].logic_expression,
                        "helper_code": planner_hints["model_agent"].helper_code,
                        "function_rewrite": planner_hints["model_agent"].function_rewrite,
                    }
                    if "model_agent" in planner_hints
                    else {}
                ),
                "augmentation_agent": (
                    {
                        "augmentation": planner_hints["augmentation_agent"].augmentation,
                        "strength": planner_hints["augmentation_agent"].strength,
                        "rationale": planner_hints["augmentation_agent"].rationale,
                        "logic_expression": planner_hints["augmentation_agent"].logic_expression,
                        "helper_code": planner_hints["augmentation_agent"].helper_code,
                        "function_rewrite": planner_hints["augmentation_agent"].function_rewrite,
                    }
                    if "augmentation_agent" in planner_hints
                    else {}
                ),
                "data_agent": (
                    {
                        "preprocessing": planner_hints["data_agent"].preprocessing,
                        "window_size": planner_hints["data_agent"].window_size,
                        "rationale": planner_hints["data_agent"].rationale,
                        "logic_expression": planner_hints["data_agent"].logic_expression,
                        "helper_code": planner_hints["data_agent"].helper_code,
                        "function_rewrite": planner_hints["data_agent"].function_rewrite,
                    }
                    if "data_agent" in planner_hints
                    else {}
                ),
            },
        }
        planner_artifact = write_json_artifact(round_dir / "planner_decision.json", planner_payload)
        memory_dir = round_dir / "agent_memories"
        memory_artifacts = {
            agent_name: write_json_artifact(memory_dir / f"{agent_name}.json", memory_payload)
            for agent_name, memory_payload in agent_memories.items()
        }

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
                "augmentation_strength": augmentation_proposal.strength,
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
            planner_artifact=planner_artifact,
            memory_artifacts=memory_artifacts,
            proposal_artifacts=proposal_artifacts,
            composed_artifact=composed_artifact,
            orchestrator_branch=self.orchestrator_workspace.branch_name,
            orchestrator_worktree=self.orchestrator_workspace.worktree_path,
        )

    def _load_json(self, path: str) -> dict[str, object]:
        import json

        return json.loads(Path(path).read_text(encoding="utf-8"))

    def _annotate_best_so_far(
        self,
        round_results: list[CandidateResult],
        best_by_agent: dict[str, float],
    ) -> None:
        for result in round_results:
            best_by_agent[result.agent_name] = max(
                best_by_agent.get(result.agent_name, float("-inf")),
                result.val_score,
            )
            result.best_so_far_score = best_by_agent[result.agent_name]

    def _select_accepted_candidate(
        self,
        round_results: list[CandidateResult],
        best_result: CandidateResult | None,
    ) -> CandidateResult | None:
        round_best = self.evaluate_round(round_results)
        if best_result is None or round_best.val_score > best_result.val_score:
            return round_best
        return None

    def _clone_result(self, result: CandidateResult) -> CandidateResult:
        return CandidateResult(
            round_id=result.round_id,
            agent_name=result.agent_name,
            branch_name=result.branch_name,
            worktree_path=result.worktree_path,
            config=result.config,
            val_score=result.val_score,
            summary=result.summary,
            score_breakdown=dict(result.score_breakdown),
            details=dict(result.details),
            component_summaries=dict(result.component_summaries),
            artifact_paths=dict(result.artifact_paths),
            accepted=result.accepted,
            best_so_far_score=result.best_so_far_score,
        )

    def _workspace_for_branch(self, branch_name: str) -> AgentWorkspace:
        if branch_name == "main":
            return AgentWorkspace(
                agent_name="main",
                branch_name="main",
                worktree_path=str(Path.cwd()),
            )
        if branch_name == self.orchestrator_workspace.branch_name:
            return self.orchestrator_workspace
        for workspace in self.workspaces.values():
            if workspace.branch_name == branch_name:
                return workspace
        raise KeyError(f"Unknown branch workspace for {branch_name}")

    def _baseline_experiment_from_branch(self, baseline_branch: str) -> dict[str, object]:
        if baseline_branch == "main":
            return base_experiment_config()
        workspace = self._workspace_for_branch(baseline_branch)
        return compose_experiment_from_workspace(
            Path(workspace.worktree_path),
            fallback_experiment=base_experiment_config(),
            apply_modules=False,
        )

    def _compose_experiment_from_proposals(
        self,
        proposals: list[ModelProposal | AugmentationProposal | DataProposal],
        baseline_experiment: dict[str, object],
        included_agents: set[str] | None = None,
    ) -> dict[str, object]:
        experiment = clone_experiment(baseline_experiment)
        included = included_agents or {
            "model_agent",
            "augmentation_agent",
            "data_agent",
        }
        if "model_agent" in included:
            model_proposal = next(
                proposal for proposal in proposals if isinstance(proposal, ModelProposal)
            )
            experiment = apply_model_proposal(experiment, model_proposal)
        if "augmentation_agent" in included:
            augmentation_proposal = next(
                proposal for proposal in proposals if isinstance(proposal, AugmentationProposal)
            )
            experiment = apply_augmentation_proposal(experiment, augmentation_proposal)
        if "data_agent" in included:
            data_proposal = next(
                proposal for proposal in proposals if isinstance(proposal, DataProposal)
            )
            experiment = apply_data_proposal(experiment, data_proposal)
        return experiment

    def _component_summaries(
        self,
        proposals: list[ModelProposal | AugmentationProposal | DataProposal],
        included_agents: set[str],
    ) -> dict[str, str]:
        summaries: dict[str, str] = {}
        for proposal in proposals:
            if isinstance(proposal, ModelProposal) and "model_agent" in included_agents:
                summaries["model_agent"] = proposal.notes
            elif isinstance(proposal, AugmentationProposal) and "augmentation_agent" in included_agents:
                summaries["augmentation_agent"] = proposal.notes
            elif isinstance(proposal, DataProposal) and "data_agent" in included_agents:
                summaries["data_agent"] = proposal.notes
        return summaries

    def _write_run_outputs(
        self,
        history: list[CandidateResult],
        accepted_history: list[CandidateResult],
    ) -> RunOutputs:
        output_dir = Path.cwd() / "artifacts" / "reports"
        trial_csv_path = write_history_csv(
            history,
            output_dir / "score_history.csv",
            history_kind="trial",
        )
        trial_svg_path = write_history_svg(
            history,
            output_dir / "score_history.svg",
            value_key="val_score",
            title="Trial Candidate Macro-F1 By Round",
            legend_labels={"orchestrator": "orchestrator_trial"},
        )
        accepted_csv_path = write_history_csv(
            accepted_history,
            output_dir / "accepted_history.csv",
            history_kind="accepted",
        )
        accepted_svg_path = write_promotion_svg(
            accepted_history,
            output_dir / "accepted_history.svg",
            title="Accepted Baseline Macro-F1 By Round",
        )
        return RunOutputs(
            trial_history_csv=trial_csv_path,
            trial_history_svg=trial_svg_path,
            accepted_history_csv=accepted_csv_path,
            accepted_history_svg=accepted_svg_path,
            latest_promotion_artifact=(
                str(Path.cwd() / "artifacts" / f"round_{history[-1].round_id}" / "promotion.json")
                if history
                else ""
            ),
        )

    def _write_promotion_artifact(
        self,
        round_id: int,
        round_results: list[CandidateResult],
        round_best: CandidateResult,
        previous_best: CandidateResult | None,
        accepted_candidate: CandidateResult | None,
        current_best: CandidateResult | None,
    ) -> str:
        orchestrator_order = next(
            (
                result.artifact_paths.get("combination_order", "")
                for result in round_results
                if result.agent_name == "orchestrator"
                and result.artifact_paths.get("combination_order", "")
            ),
            "",
        )
        previous_baseline_score = None if previous_best is None else float(previous_best.val_score)
        payload = {
            "round_id": round_id,
            "orchestrator_combination_order": (
                orchestrator_order.split(",") if orchestrator_order else []
            ),
            "previous_baseline": None
            if previous_best is None
            else {
                "agent_name": previous_best.agent_name,
                "branch_name": previous_best.branch_name,
                "val_score": round(previous_best.val_score, 4),
            },
            "round_winner": {
                "agent_name": round_best.agent_name,
                "branch_name": round_best.branch_name,
                "val_score": round(round_best.val_score, 4),
                "selected_combination": round_best.artifact_paths.get("selected_combination", ""),
            },
            "accepted_new_baseline": accepted_candidate is not None,
            "promoted_baseline": None
            if current_best is None
            else {
                "agent_name": current_best.agent_name,
                "branch_name": current_best.branch_name,
                "val_score": round(current_best.val_score, 4),
            },
            "rejected_because_not_better_than_baseline": (
                accepted_candidate is None and previous_best is not None
            ),
            "candidates": [
                self._promotion_candidate_payload(
                    result=result,
                    round_best=round_best,
                    previous_baseline_score=previous_baseline_score,
                    accepted_candidate=accepted_candidate,
                )
                for result in sorted(round_results, key=lambda item: item.val_score, reverse=True)
            ],
        }
        round_dir = ensure_round_dir(round_id)
        return write_json_artifact(round_dir / "promotion.json", payload)

    def _promotion_candidate_payload(
        self,
        result: CandidateResult,
        round_best: CandidateResult,
        previous_baseline_score: float | None,
        accepted_candidate: CandidateResult | None,
    ) -> dict[str, object]:
        selected_combination = result.artifact_paths.get("selected_combination", "")
        is_round_winner = (
            result.agent_name == round_best.agent_name
            and result.branch_name == round_best.branch_name
            and abs(result.val_score - round_best.val_score) < 1e-9
        )
        beats_previous_baseline = (
            previous_baseline_score is None or result.val_score > previous_baseline_score
        )
        accepted_as_new_baseline = (
            accepted_candidate is not None
            and result.agent_name == accepted_candidate.agent_name
            and result.branch_name == accepted_candidate.branch_name
            and abs(result.val_score - accepted_candidate.val_score) < 1e-9
        )
        if result.agent_name == "model_agent":
            domain_focus = "model"
        elif result.agent_name == "augmentation_agent":
            domain_focus = "augmentation"
        elif result.agent_name == "data_agent":
            domain_focus = "data"
        else:
            domain_focus = f"composition:{selected_combination or 'orchestrator'}"

        if accepted_as_new_baseline:
            decision = "promoted"
        elif not beats_previous_baseline:
            decision = "rejected_below_previous_baseline"
        elif not is_round_winner:
            decision = "rejected_not_best_in_round"
        else:
            decision = "round_winner_but_not_promoted"

        return {
            "agent_name": result.agent_name,
            "branch_name": result.branch_name,
            "domain_focus": domain_focus,
            "candidate_type": (
                "orchestrator_combination" if result.agent_name == "orchestrator" else "specialist_agent"
            ),
            "val_score": round(result.val_score, 4),
            "selected_combination": selected_combination,
            "is_round_winner": is_round_winner,
            "beats_previous_baseline": beats_previous_baseline,
            "accepted_as_new_baseline": accepted_as_new_baseline,
            "decision": decision,
            "summary": result.summary,
            "proposal_summary": result.config.notes,
            "component_summaries": dict(result.component_summaries),
        }

    def _write_result_artifact(self, result: CandidateResult) -> str:
        round_dir = ensure_round_dir(result.round_id)
        return write_json_artifact(
            round_dir / "results" / f"{result.agent_name}.json",
            {
                "round_id": result.round_id,
                "agent_name": result.agent_name,
                "branch_name": result.branch_name,
                "summary": result.summary,
                "config": {
                    "model_name": result.config.model_name,
                    "model_params": result.config.model_params,
                    "preprocessing": result.config.preprocessing,
                    "data_params": result.config.data_params,
                    "augmentation": result.config.augmentation,
                    "augmentation_strength": result.config.augmentation_strength,
                    "augmentation_params": result.config.augmentation_params,
                    "learning_rate": result.config.learning_rate,
                    "window_size": result.config.window_size,
                    "notes": result.config.notes,
                },
                "score_breakdown": result.score_breakdown,
                "details": result.details,
            },
        )

    def finalize_actions(self, best_result: CandidateResult) -> list[GitAction]:
        return [
            GitAction(
                label="Promote best accepted branch",
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
