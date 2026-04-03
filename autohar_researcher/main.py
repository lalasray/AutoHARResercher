from __future__ import annotations

import argparse

from autohar_researcher.agents import HierarchicalOrchestrator
from autohar_researcher.harness import DummyHARHarness


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
    parser = argparse.ArgumentParser(description="Minimal multi-agent HAR auto-research demo.")
    parser.add_argument("--rounds", type=int, default=3, help="How many orchestrator rounds to run.")
    parser.add_argument(
        "--show-git-plan",
        action="store_true",
        help="Print the git-first orchestration plan for branches and worktrees.",
    )
    args = parser.parse_args()

    orchestrator = HierarchicalOrchestrator(harness=DummyHARHarness())
    results = orchestrator.run(rounds=args.rounds)

    history = results["history"]
    best_result = results["best_result"]
    workspaces = results["workspaces"]
    git_actions = results["git_actions"]
    ranked = sorted(history, key=lambda item: item.val_score, reverse=True)

    print("AutoHARResearcher git-first demo")
    print("=" * 32)
    print(f"Rounds: {args.rounds}")
    print(f"Candidates evaluated: {len(history)}")
    print()
    print("Agent workspaces:")
    for workspace in workspaces.values():
        print(
            f"- {workspace.agent_name}: branch={workspace.branch_name} "
            f"worktree={workspace.worktree_path}"
        )
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
    print("Score breakdown:")
    for key, value in best_result.score_breakdown.items():
        print(f"- {key}: {value:.4f}")

    if args.show_git_plan:
        print()
        print("Git orchestration plan:")
        for action in git_actions:
            print(f"- {action.label}: {action.command}")


if __name__ == "__main__":
    main()
