from __future__ import annotations

from autohar_researcher.agents import HierarchicalOrchestrator
from autohar_researcher.types import CandidateConfig, CandidateResult


def _result(
    agent_name: str,
    score: float,
    *,
    crash: bool = False,
    timeout: bool = False,
) -> CandidateResult:
    breakdown = {"macro_f1": score}
    details = {}
    summary = "ok"
    if crash:
        breakdown["crash"] = 1.0
        details["crash"] = True
        summary = "Crash: boom"
    if timeout:
        breakdown["timeout"] = 1.0
        details["timeout"] = True
        summary = "Timeout"
    return CandidateResult(
        round_id=1,
        agent_name=agent_name,
        branch_name=f"{agent_name}-branch",
        worktree_path="",
        config=CandidateConfig(
            model_name="cnn",
            preprocessing="raw",
            augmentation="none",
            augmentation_strength="medium",
            learning_rate=0.001,
            window_size=64,
        ),
        val_score=score,
        summary=summary,
        score_breakdown=breakdown,
        details=details,
    )


def test_crashed_candidate_is_not_accepted_without_baseline():
    orchestrator = object.__new__(HierarchicalOrchestrator)
    crashed = _result("model_agent", 0.0, crash=True)

    assert orchestrator._select_accepted_candidate([crashed], None) is None


def test_evaluate_round_prefers_successful_result_over_crashed_score_tie():
    orchestrator = object.__new__(HierarchicalOrchestrator)
    crashed = _result("model_agent", 0.0, crash=True)
    successful = _result("data_agent", 0.0)

    assert orchestrator.evaluate_round([crashed, successful]) is successful


def test_promotion_payload_marks_crash_rejected():
    orchestrator = object.__new__(HierarchicalOrchestrator)
    crashed = _result("model_agent", 0.0, crash=True)

    payload = orchestrator._promotion_candidate_payload(
        result=crashed,
        round_best=crashed,
        previous_baseline_score=None,
        accepted_candidate=None,
    )

    assert payload["beats_previous_baseline"] is False
    assert payload["accepted_as_new_baseline"] is False
    assert payload["decision"] == "rejected_crash_or_timeout"
