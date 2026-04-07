"""
Simple autoresearch loop - runs train.py repeatedly and tracks results.

Inspired by karpathy/autoresearch:
1. Agent modifies research/train.py
2. Run the experiment
3. Check results
4. Keep if improved, revert if worse
5. Repeat
"""

from __future__ import annotations

import json
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from autohar_researcher.results_tracker import (
    ExperimentResult,
    ResultsTracker,
    create_results_tracker,
    get_git_commit,
)


@dataclass
class AutoresearchLoop:
    repo_root: Path
    train_script: Path
    agent_instructions: Path
    results_tracker: ResultsTracker
    max_time_budget: float | None = None
    verbose: bool = True

    def __post_init__(self) -> None:
        self.experiment_count = 0
        self.best_score = None
        self.best_commit = None

    def run(self, max_experiments: int | None = None) -> None:
        print("=" * 60)
        print("AutoHARResearcher - Autonomous HAR Research")
        print("=" * 60)
        print(f"Train script: {self.train_script}")
        print(f"Results: {self.results_tracker.path}")
        print()

        while True:
            if max_experiments and self.experiment_count >= max_experiments:
                print(f"\nReached max experiments ({max_experiments})")
                break

            try:
                self.run_experiment()
            except KeyboardInterrupt:
                print("\n\nInterrupted by user")
                break
            except Exception as e:
                print(f"\nExperiment failed: {e}")
                self.record_crash(str(e))

            self.experiment_count += 1

    def run_experiment(self) -> float | None:
        commit_before = get_git_commit()

        if self.verbose:
            print(f"\n--- Experiment {self.experiment_count + 1} ---")
            print(f"Commit: {commit_before[:8]}")

        start_time = time.time()

        result = subprocess.run(
            ["python", str(self.train_script)],
            capture_output=True,
            text=True,
            cwd=self.repo_root,
            timeout=self.max_time_budget,
        )

        elapsed = time.time() - start_time

        if result.returncode != 0:
            print(f"  FAILED (exit code {result.returncode})")
            if self.verbose:
                print(f"  Error: {result.stderr[:200]}")
            self.record_crash(result.stderr[:200])
            return None

        score = self.parse_score(result.stdout)
        if score is None:
            print(f"  FAILED (could not parse score)")
            self.record_crash("Could not parse score from output")
            return None

        if self.verbose:
            print(f"  Score: {score:.4f}")
            print(f"  Time: {elapsed:.1f}s")

        commit_after = get_git_commit()
        changed = commit_before != commit_after

        status = self.decide_status(score, changed)
        self.record_result(score, status, elapsed, commit_before)

        if status == "keep":
            self.best_score = score
            self.best_commit = commit_before
            if self.verbose:
                print(f"  NEW BEST: {score:.4f}")
        elif self.verbose:
            print(f"  Status: {status}")

        return score

    def parse_score(self, output: str) -> float | None:
        for line in output.split("\n"):
            if "macro-f1" in line.lower():
                parts = line.split()
                for i, part in enumerate(parts):
                    if "macro-f1" in part.lower():
                        try:
                            return float(parts[i + 1])
                        except (IndexError, ValueError):
                            pass
            if line.startswith("Macro-F1:"):
                try:
                    return float(line.split(":")[1].strip())
                except (IndexError, ValueError):
                    pass
            if "val_score:" in line:
                try:
                    return float(line.split(":")[1].strip())
                except (IndexError, ValueError):
                    pass
        return None

    def decide_status(self, score: float, changed: bool) -> str:
        if self.best_score is None:
            return "keep"
        if score > self.best_score:
            return "keep"
        if abs(score - self.best_score) < 0.001:
            return "keep"  # simplification
        return "discard"

    def record_result(
        self,
        score: float,
        status: str,
        elapsed: float,
        commit: str,
    ) -> None:
        result = ExperimentResult(
            commit=commit,
            val_score=score,
            memory_gb=0.0,
            status=status,
            description=f"experiment_{self.experiment_count}",
            training_seconds=elapsed,
            total_seconds=elapsed,
            domain="autoresearch",
        )
        self.results_tracker.add(result)

    def record_crash(self, error: str) -> None:
        result = ExperimentResult(
            commit=get_git_commit(),
            val_score=0.0,
            memory_gb=0.0,
            status="crash",
            description=error[:200],
            domain="autoresearch",
        )
        self.results_tracker.add(result)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="AutoHARResearcher loop")
    parser.add_argument(
        "--repo-root",
        type=str,
        default=".",
        help="Repository root directory",
    )
    parser.add_argument(
        "--max-experiments",
        type=int,
        default=None,
        help="Maximum number of experiments (default: unlimited)",
    )
    parser.add_argument(
        "--time-budget",
        type=float,
        default=None,
        help="Time budget per experiment in seconds",
    )
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    train_script = repo_root / "research" / "train.py"
    agent_instructions = repo_root / "agent.md"
    results_path = repo_root / "results.tsv"

    if not train_script.exists():
        print(f"Error: {train_script} not found")
        print("Create research/train.py with your training code")
        return 1

    tracker = create_results_tracker(results_path, repo_root)

    loop = AutoresearchLoop(
        repo_root=repo_root,
        train_script=train_script,
        agent_instructions=agent_instructions,
        results_tracker=tracker,
        max_time_budget=args.time_budget,
    )

    loop.run(max_experiments=args.max_experiments)

    print("\n" + "=" * 60)
    print("Final Results")
    print("=" * 60)
    print(f"Experiments: {loop.experiment_count}")
    print(
        f"Best score: {loop.best_score:.4f}"
        if loop.best_score
        else "No successful experiments"
    )
    print(f"Results saved to: {results_path}")

    return 0


if __name__ == "__main__":
    exit(main())
