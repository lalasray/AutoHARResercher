from __future__ import annotations

import argparse

from autohar_researcher.agents import HierarchicalOrchestrator
from autohar_researcher.config import load_autoresearch_config, load_time_budget_config
from autohar_researcher.harness import DummyHARHarness
from autohar_researcher.time_budget import create_timed_harness


def format_result_line(
    rank: int,
    round_id: int,
    agent_name: str,
    branch_name: str,
    score: float,
    summary: str,
) -> str:
    return (
        f"{rank}. round={round_id} agent={agent_name} branch={branch_name} "
        f"val_score={score:.4f} | {summary}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Minimal multi-agent HAR auto-research demo."
    )
    parser.add_argument(
        "--rounds", type=int, default=3, help="How many orchestrator rounds to run."
    )
    parser.add_argument(
        "--show-git-plan",
        action="store_true",
        help="Print the git-first orchestration plan for branches and worktrees.",
    )
    parser.add_argument(
        "--materialize-git",
        action="store_true",
        help="Actually create/update agent worktrees, write files, and commit round changes.",
    )
    parser.add_argument(
        "--auto-push-non-main",
        action="store_true",
        help="Push agent branches and orchestrator-main to origin after commit, but never push main.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from the latest completed round on disk and continue with additional rounds.",
    )
    parser.add_argument(
        "--never-stop",
        action="store_true",
        help="Run indefinitely until manually stopped. Ignores --rounds.",
    )
    parser.add_argument(
        "--max-rounds-without-improvement",
        type=int,
        default=10,
        help="Stop never-stop mode after this many rounds without improvement.",
    )
    parser.add_argument(
        "--results-path",
        type=str,
        default=None,
        help="Path for results.tsv tracking file (autoresearch mode).",
    )
    parser.add_argument(
        "--time-budget",
        type=float,
        default=None,
        help="Time budget per experiment in seconds (like karpathy/autoresearch).",
    )
    args = parser.parse_args()

    autoresearch_config = load_autoresearch_config()
    if args.never_stop:
        import os

        os.environ["AUTOHAR_NEVER_STOP"] = "true"
        autoresearch_config = load_autoresearch_config()

    time_budget_config = load_time_budget_config()
    if args.time_budget is not None:
        from autohar_researcher.config import TimeBudgetConfig

        time_budget_config = TimeBudgetConfig(
            enabled=True,
            budget_seconds=args.time_budget,
            startup_overhead_seconds=time_budget_config.startup_overhead_seconds,
        )

    base_harness = DummyHARHarness()
    harness = create_timed_harness(base_harness, time_budget_config)

    orchestrator = HierarchicalOrchestrator(
        harness=harness,
        autoresearch_config=autoresearch_config,
        time_budget_config=time_budget_config,
    )

    if args.never_stop or autoresearch_config.never_stop:
        print("[AutoHAR] Running in never-stop (autoresearch) mode", flush=True)
        results = orchestrator.run_never_stop(
            materialize_git=args.materialize_git,
            auto_push_non_main=args.auto_push_non_main,
            resume=args.resume,
            max_rounds_without_improvement=args.max_rounds_without_improvement,
            results_path=args.results_path,
        )
    else:
        results = orchestrator.run(
            rounds=args.rounds,
            materialize_git=args.materialize_git,
            auto_push_non_main=args.auto_push_non_main,
            resume=args.resume,
        )

    history = results["history"]
    best_result = results["best_result"]
    workspaces = results["workspaces"]
    git_actions = results["git_actions"]
    round_states = results["round_states"]
    run_outputs = results.get("run_outputs")
    planner_name = results["planner_name"]
    ranked = sorted(history, key=lambda item: item.val_score, reverse=True)

    print("AutoHARResearcher git-first demo")
    print("=" * 32)
    print(f"Rounds: {args.rounds if not args.never_stop else 'never-stop'}")
    print(f"Candidates evaluated: {len(history)}")
    print(f"Materialize git: {args.materialize_git}")
    print(f"Auto-push non-main: {args.auto_push_non_main}")
    print(f"Resume mode: {args.resume}")
    print(f"Planner: {planner_name}")
    if args.never_stop or autoresearch_config.never_stop:
        experiment_count = results.get("experiment_count", 0)
        print(f"Experiments completed: {experiment_count}")
        results_tracker = results.get("results_tracker")
        if results_tracker:
            print(f"Results tracker: {results_tracker.path}")
    if time_budget_config.enabled:
        print(f"Time budget: {time_budget_config.budget_seconds}s per experiment")
    print(f"Training backend: {orchestrator.harness.runtime_backend_summary()}")
    role_summary = getattr(orchestrator.planner, "role_runtime_summary", lambda: {})()
    if role_summary:
        print("Planner roles:")
        for role_name, payload in role_summary.items():
            backend = payload.get("backend", "")
            model = payload.get("model", "")
            api_mode = payload.get("api_mode", "")
            base_url = payload.get("base_url", "")
            extra = f" base_url={base_url}" if base_url else ""
            print(
                f"- {role_name}: backend={backend} model={model} api_mode={api_mode}{extra}"
            )
    print()
    print("Agent workspaces:")
    for workspace in workspaces.values():
        print(
            f"- {workspace.agent_name}: branch={workspace.branch_name} "
            f"worktree={workspace.worktree_path}"
        )
    print(
        f"- orchestrator: branch=orchestrator-main "
        f"worktree={orchestrator.orchestrator_workspace.worktree_path}"
    )
    print()
    print("Round artifacts:")
    for round_state in round_states:
        print(f"- round {round_state.round_id}:")
        if round_state.planner_artifact:
            print(f"  planner decision -> {round_state.planner_artifact}")
        if round_state.sync_artifact:
            print(f"  baseline sync -> {round_state.sync_artifact}")
        for agent_name, artifact_path in round_state.memory_artifacts.items():
            print(f"  {agent_name} memory -> {artifact_path}")
        for agent_name, artifact_path in round_state.proposal_artifacts.items():
            print(f"  {agent_name} proposal -> {artifact_path}")
        print(f"  orchestrator candidate -> {round_state.composed_artifact}")
        if round_state.promotion_artifact:
            print(f"  promotion decision -> {round_state.promotion_artifact}")
        if round_state.synced_branches:
            print(f"  synced branches -> {', '.join(round_state.synced_branches)}")
        if round_state.pushed_branches:
            print(f"  pushed branches -> {', '.join(round_state.pushed_branches)}")
    print()
    print("Report outputs:")
    print(f"- trial history csv: {run_outputs.trial_history_csv}")
    print(f"- trial history svg: {run_outputs.trial_history_svg}")
    print(f"- accepted history csv: {run_outputs.accepted_history_csv}")
    print(f"- accepted history svg: {run_outputs.accepted_history_svg}")
    if run_outputs.latest_promotion_artifact:
        print(f"- latest promotion artifact: {run_outputs.latest_promotion_artifact}")
    print()
    print("Top candidates:")
    for rank, result in enumerate(ranked[:5], start=1):
        print(
            format_result_line(
                rank=rank,
                round_id=result.round_id,
                agent_name=result.agent_name,
                branch_name=result.branch_name,
                score=result.val_score,
                summary=result.summary,
            )
        )

    print()
    print("Best configuration kept by orchestrator:")
    print(best_result.config)
    print(f"Best branch to promote: {best_result.branch_name}")
    print(f"Best worktree: {best_result.worktree_path}")
    if best_result.component_summaries:
        print("Composed from agent proposals:")
        for agent_name, summary in best_result.component_summaries.items():
            print(f"- {agent_name}: {summary}")
    if best_result.artifact_paths:
        print("Related artifacts:")
        for label, artifact_path in best_result.artifact_paths.items():
            print(f"- {label}: {artifact_path}")
    print("Score breakdown:")
    for key, value in best_result.score_breakdown.items():
        print(f"- {key}: {value:.4f}")
    if best_result.details.get("per_class_f1"):
        print("Per-class F1:")
        for label, score in best_result.details["per_class_f1"].items():
            print(f"- {label}: {score:.4f}")
    if best_result.details.get("confusion_matrix"):
        print("Confusion matrix:")
        labels = best_result.details.get("labels", [])
        if labels:
            print(f"- labels: {', '.join(labels)}")
        for row in best_result.details["confusion_matrix"]:
            print(f"- {' '.join(str(value) for value in row)}")
    if best_result.details.get("training_report"):
        print("Training report:")
        for key, value in best_result.details["training_report"].items():
            print(f"- {key}: {value}")

    if args.show_git_plan:
        print()
        print("Git orchestration plan:")
        for action in git_actions:
            print(f"- {action.label}: {action.command}")


if __name__ == "__main__":
    main()
