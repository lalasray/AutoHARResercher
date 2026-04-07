from __future__ import annotations

import os
import signal
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from autohar_researcher.types import CandidateConfig, CandidateResult


@dataclass
class TimeBudget:
    seconds: float
    startup_overhead: float = 10.0

    def effective_budget(self) -> float:
        return max(0.0, self.seconds - self.startup_overhead)

    def is_expired(self, elapsed: float) -> bool:
        return elapsed >= self.seconds


class TimeoutException(Exception):
    pass


def _timeout_handler(signum: int, frame) -> None:
    raise TimeoutException("Training exceeded time budget")


@dataclass
class TimedEvaluationResult:
    val_score: float
    training_seconds: float
    total_seconds: float
    peak_vram_mb: float
    timeout: bool = False
    crashed: bool = False
    error_message: str = ""


@dataclass
class TimedHarnessWrapper:
    harness: object
    time_budget: TimeBudget | None = None
    _start_time: float = field(default=0.0, init=False, repr=False)

    def __post_init__(self) -> None:
        self._start_time = 0.0

    def runtime_backend_summary(self) -> str:
        return self.harness.runtime_backend_summary()

    def evaluate(
        self,
        round_id: int,
        agent_name: str,
        branch_name: str,
        worktree_path: str,
        config: "CandidateConfig",
        **kwargs,
    ) -> "CandidateResult":
        if self.time_budget is None:
            return self.harness.evaluate(
                round_id=round_id,
                agent_name=agent_name,
                branch_name=branch_name,
                worktree_path=worktree_path,
                config=config,
                **kwargs,
            )

        self._start_time = time.time()
        timeout_seconds = int(self.time_budget.seconds)

        old_handler = None
        if hasattr(signal, "SIGALRM"):
            old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
            signal.alarm(timeout_seconds)

        try:
            result = self.harness.evaluate(
                round_id=round_id,
                agent_name=agent_name,
                branch_name=branch_name,
                worktree_path=worktree_path,
                config=config,
                **kwargs,
            )
            total_elapsed = time.time() - self._start_time

            if hasattr(signal, "SIGALRM"):
                signal.alarm(0)

            return result

        except TimeoutException:
            if hasattr(signal, "SIGALRM"):
                signal.alarm(0)

            return self._create_timeout_result(
                round_id=round_id,
                agent_name=agent_name,
                branch_name=branch_name,
                worktree_path=worktree_path,
                config=config,
            )

        except Exception as e:
            if hasattr(signal, "SIGALRM"):
                signal.alarm(0)

            return self._create_crash_result(
                round_id=round_id,
                agent_name=agent_name,
                branch_name=branch_name,
                worktree_path=worktree_path,
                config=config,
                error=str(e),
            )

        finally:
            if old_handler is not None and hasattr(signal, "SIGALRM"):
                signal.signal(signal.SIGALRM, old_handler)

    def _create_timeout_result(
        self,
        round_id: int,
        agent_name: str,
        branch_name: str,
        worktree_path: str,
        config: "CandidateConfig",
    ) -> "CandidateResult":
        from autohar_researcher.types import CandidateConfig, CandidateResult

        elapsed = time.time() - self._start_time

        return CandidateResult(
            round_id=round_id,
            agent_name=agent_name,
            branch_name=branch_name,
            worktree_path=worktree_path,
            config=config,
            val_score=0.0,
            summary=f"Timeout after {elapsed:.1f}s (budget: {self.time_budget.seconds}s)",
            score_breakdown={"timeout": 1.0, "macro_f1": 0.0},
            details={
                "timeout": True,
                "elapsed_seconds": round(elapsed, 1),
                "time_budget_seconds": self.time_budget.seconds,
                "training_seconds": 0.0,
                "total_seconds": round(elapsed, 1),
            },
        )

    def _create_crash_result(
        self,
        round_id: int,
        agent_name: str,
        branch_name: str,
        worktree_path: str,
        config: "CandidateConfig",
        error: str,
    ) -> "CandidateResult":
        from autohar_researcher.types import CandidateConfig, CandidateResult

        elapsed = time.time() - self._start_time

        return CandidateResult(
            round_id=round_id,
            agent_name=agent_name,
            branch_name=branch_name,
            worktree_path=worktree_path,
            config=config,
            val_score=0.0,
            summary=f"Crash: {error[:200]}",
            score_breakdown={"crash": 1.0, "macro_f1": 0.0},
            details={
                "crash": True,
                "error": error,
                "elapsed_seconds": round(elapsed, 1),
                "training_seconds": 0.0,
                "total_seconds": round(elapsed, 1),
            },
        )


@dataclass
class HarnessTimeBudgetConfig:
    enabled: bool = False
    budget_seconds: float = 300.0
    startup_overhead_seconds: float = 10.0

    @classmethod
    def from_env(cls) -> "HarnessTimeBudgetConfig":
        enabled = os.getenv("AUTOHAR_TIME_BUDGET_ENABLED", "false").lower() == "true"
        if not enabled:
            return cls(enabled=False)

        return cls(
            enabled=True,
            budget_seconds=float(os.getenv("AUTOHAR_TIME_BUDGET_SECONDS", "300")),
            startup_overhead_seconds=float(
                os.getenv("AUTOHAR_TIME_BUDGET_STARTUP_OVERHEAD", "10")
            ),
        )


def create_timed_harness(
    harness,
    config: HarnessTimeBudgetConfig | None = None,
) -> TimedHarnessWrapper:
    if config is None:
        config = HarnessTimeBudgetConfig.from_env()

    if not config.enabled:
        return TimedHarnessWrapper(harness=harness, time_budget=None)

    time_budget = TimeBudget(
        seconds=config.budget_seconds,
        startup_overhead=config.startup_overhead_seconds,
    )

    return TimedHarnessWrapper(harness=harness, time_budget=time_budget)
