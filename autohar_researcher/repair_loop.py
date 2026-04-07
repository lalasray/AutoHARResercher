from __future__ import annotations

import json
import time
import traceback
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from autohar_researcher.artifacts import ensure_round_dir, write_json_artifact
from autohar_researcher.config import RepairConfig, load_repair_config
from autohar_researcher.types import (
    AugmentationProposal,
    AugmentationPlanHint,
    CandidateResult,
    DataProposal,
    DataPlanHint,
    FilePatch,
    ModelProposal,
    ModelPlanHint,
    ProposalJudgment,
)

if TYPE_CHECKING:
    from autohar_researcher.heuristic_planner import HeuristicPlanner
    from autohar_researcher.planner import OpenAIPlanner


@dataclass
class RepairAttempt:
    attempt: int
    error_summary: str
    repair_strategy: str
    success: bool
    repaired_proposals: (
        list[ModelProposal | AugmentationProposal | DataProposal] | None
    ) = None
    repair_artifact: str = ""


@dataclass
class RepairRecord:
    round_id: int
    failure_type: str
    error_traceback: str
    attempts: list[RepairAttempt] = field(default_factory=list)
    final_success: bool = False
    fell_back_to_heuristic: bool = False
    record_artifact: str = ""


def _safe_traceback() -> str:
    try:
        return traceback.format_exc()
    except Exception:
        return "Traceback unavailable"


def _summarize_error(exc: Exception) -> str:
    return f"{type(exc).__name__}: {exc}"


def _error_context_dict(
    round_id: int,
    proposals: list[ModelProposal | AugmentationProposal | DataProposal] | None,
    round_plan: dict[str, object] | None,
    exc: Exception,
    tb: str,
    best_result: CandidateResult | None,
    history: list[CandidateResult],
) -> dict[str, object]:
    proposal_dicts: list[dict[str, object]] = []
    for p in proposals or []:
        if isinstance(p, ModelProposal):
            proposal_dicts.append(
                {
                    "agent": "model_agent",
                    "model_name": p.model_name,
                    "learning_rate": p.learning_rate,
                    "notes": p.notes,
                }
            )
        elif isinstance(p, AugmentationProposal):
            proposal_dicts.append(
                {
                    "agent": "augmentation_agent",
                    "augmentation": p.augmentation,
                    "strength": p.strength,
                    "notes": p.notes,
                }
            )
        else:
            proposal_dicts.append(
                {
                    "agent": "data_agent",
                    "preprocessing": p.preprocessing,
                    "window_size": p.window_size,
                    "notes": p.notes,
                }
            )

    recent = [
        {
            "round_id": r.round_id,
            "agent": r.agent_name,
            "score": r.val_score,
            "config_model": r.config.model_name,
        }
        for r in history[-6:]
    ]

    return {
        "round_id": round_id,
        "error_type": type(exc).__name__,
        "error_message": str(exc),
        "traceback": tb,
        "proposals": proposal_dicts,
        "round_plan_summary": {
            "planner": str(round_plan.get("planner_name", ""))
            if round_plan
            else "unknown",
            "baseline_branch": str(round_plan.get("baseline_branch", ""))
            if round_plan
            else "",
        }
        if round_plan
        else {},
        "best_result": {
            "score": best_result.val_score,
            "model": best_result.config.model_name,
        }
        if best_result
        else None,
        "recent_history": recent,
    }


class CrashRepairLoop:
    def __init__(
        self,
        planner: OpenAIPlanner | HeuristicPlanner,
        config: RepairConfig | None = None,
    ) -> None:
        self.planner = planner
        self.config = config or load_repair_config()

    def execute_with_repair(
        self,
        round_id: int,
        execute_fn,
        *,
        best_result: CandidateResult | None = None,
        history: list[CandidateResult] | None = None,
        round_plan: dict[str, object] | None = None,
        proposals: list[ModelProposal | AugmentationProposal | DataProposal]
        | None = None,
    ):
        history = history or []
        record = RepairRecord(
            round_id=round_id,
            failure_type="",
            error_traceback="",
        )

        for attempt_num in range(1, self.config.max_retries + 1):
            try:
                result = execute_fn()
                if attempt_num > 1:
                    record.final_success = True
                    record.record_artifact = self._write_repair_record(round_id, record)
                return result
            except Exception as exc:
                tb = _safe_traceback()
                error_summary = _summarize_error(exc)

                if attempt_num == 1:
                    record.failure_type = type(exc).__name__
                    record.error_traceback = tb

                repair_strategy = self._determine_repair_strategy(
                    exc, attempt_num, proposals
                )

                attempt = RepairAttempt(
                    attempt=attempt_num,
                    error_summary=error_summary,
                    repair_strategy=repair_strategy,
                    success=False,
                )

                if attempt_num < self.config.max_retries:
                    repaired = self._attempt_repair(
                        round_id=round_id,
                        attempt_num=attempt_num,
                        exc=exc,
                        tb=tb,
                        proposals=proposals,
                        round_plan=round_plan,
                        best_result=best_result,
                        history=history,
                    )
                    if repaired is not None:
                        proposals = repaired
                        attempt.repaired_proposals = repaired
                        attempt.success = True
                        attempt.repair_artifact = self._write_repair_artifact(
                            round_id, attempt_num, repaired
                        )
                    record.attempts.append(attempt)

                    backoff = self.config.backoff_base ** (attempt_num - 1)
                    time.sleep(min(backoff, 10.0))
                    continue

                attempt.success = False
                record.attempts.append(attempt)

                if self.config.fallback_to_heuristic:
                    heuristic_repaired = self._heuristic_repair(proposals, exc)
                    if heuristic_repaired is not None:
                        proposals = heuristic_repaired
                        record.fell_back_to_heuristic = True
                        attempt.repaired_proposals = heuristic_repaired
                        attempt.repair_artifact = self._write_repair_artifact(
                            round_id, attempt_num, heuristic_repaired
                        )

                record.record_artifact = self._write_repair_record(round_id, record)
                raise

    def _determine_repair_strategy(
        self,
        exc: Exception,
        attempt_num: int,
        proposals: list[ModelProposal | AugmentationProposal | DataProposal] | None,
    ) -> str:
        error_type = type(exc).__name__
        if "JSON" in error_type or "json" in str(exc).lower():
            return "parse_error_regenerate"
        if "timeout" in error_type.lower() or "timed out" in str(exc).lower():
            return "timeout_simplify"
        if "FilePatch" in str(exc) or "patch" in str(exc).lower():
            return "patch_failure_strip_patches"
        if attempt_num >= 2:
            return "aggressive_simplify"
        return "conservative_regenerate"

    def _attempt_repair(
        self,
        round_id: int,
        attempt_num: int,
        exc: Exception,
        tb: str,
        proposals: list[ModelProposal | AugmentationProposal | DataProposal] | None,
        round_plan: dict[str, object] | None,
        best_result: CandidateResult | None,
        history: list[CandidateResult],
    ) -> list[ModelProposal | AugmentationProposal | DataProposal] | None:
        if not hasattr(self.planner, "_call_json_schema"):
            return None

        context = _error_context_dict(
            round_id, proposals, round_plan, exc, tb, best_result, history
        )

        strategy = self._determine_repair_strategy(exc, attempt_num, proposals)

        repaired = []
        for proposal in proposals or []:
            if isinstance(proposal, ModelProposal):
                hint = self._repair_model_proposal(
                    context, proposal, strategy, attempt_num
                )
                if hint is not None:
                    repaired.append(
                        ModelProposal(
                            model_name=hint.model_name,
                            learning_rate=hint.learning_rate,
                            model_params=dict(hint.model_params),
                            notes=f"[repaired attempt {attempt_num}] {hint.rationale}",
                            logic_expression=hint.logic_expression
                            if strategy != "patch_failure_strip_patches"
                            else "",
                            helper_code=hint.helper_code
                            if strategy != "patch_failure_strip_patches"
                            else "",
                            function_rewrite=hint.function_rewrite
                            if strategy != "patch_failure_strip_patches"
                            else "",
                            file_patches=hint.file_patches
                            if strategy != "patch_failure_strip_patches"
                            else [],
                        )
                    )
                else:
                    return None
            elif isinstance(proposal, AugmentationProposal):
                hint = self._repair_augmentation_proposal(
                    context, proposal, strategy, attempt_num
                )
                if hint is not None:
                    repaired.append(
                        AugmentationProposal(
                            augmentation=hint.augmentation,
                            strength=hint.strength,
                            notes=f"[repaired attempt {attempt_num}] {hint.rationale}",
                            logic_expression=hint.logic_expression
                            if strategy != "patch_failure_strip_patches"
                            else "",
                            helper_code=hint.helper_code
                            if strategy != "patch_failure_strip_patches"
                            else "",
                            function_rewrite=hint.function_rewrite
                            if strategy != "patch_failure_strip_patches"
                            else "",
                            file_patches=hint.file_patches
                            if strategy != "patch_failure_strip_patches"
                            else [],
                        )
                    )
                else:
                    return None
            else:
                hint = self._repair_data_proposal(
                    context, proposal, strategy, attempt_num
                )
                if hint is not None:
                    repaired.append(
                        DataProposal(
                            preprocessing=hint.preprocessing,
                            window_size=hint.window_size,
                            notes=f"[repaired attempt {attempt_num}] {hint.rationale}",
                            logic_expression=hint.logic_expression
                            if strategy != "patch_failure_strip_patches"
                            else "",
                            helper_code=hint.helper_code
                            if strategy != "patch_failure_strip_patches"
                            else "",
                            function_rewrite=hint.function_rewrite
                            if strategy != "patch_failure_strip_patches"
                            else "",
                            file_patches=hint.file_patches
                            if strategy != "patch_failure_strip_patches"
                            else [],
                        )
                    )
                else:
                    return None

        return repaired

    def _repair_model_proposal(
        self,
        context: dict[str, object],
        original: ModelProposal,
        strategy: str,
        attempt_num: int,
    ) -> ModelPlanHint | None:
        planner = self.planner
        if not hasattr(planner, "_call_json_schema"):
            return None

        prompt = (
            f"You are repairing a failed model proposal (attempt {attempt_num}, strategy: {strategy}).\n\n"
            f"Error context:\n{json.dumps(context, indent=2)}\n\n"
            f"Original proposal: model={original.model_name}, lr={original.learning_rate}\n\n"
            f"Return a simplified, safer model proposal that avoids the previous error.\n"
            f"Keep the model family stable if the error was a parse or patch issue.\n"
            f"Use conservative learning rates (1e-4 to 1e-3).\n"
        )
        schema = {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "model_name": {"type": "string"},
                "learning_rate": {"type": "number"},
                "model_params": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "hidden_dim": {"type": "integer"},
                        "depth": {"type": "integer"},
                        "temporal_pool": {"type": "string"},
                        "spectral_bins": {"type": "integer"},
                        "use_interactions": {"type": "boolean"},
                        "use_multiscale": {"type": "boolean"},
                        "attention_mix": {"type": "number"},
                    },
                    "required": [
                        "hidden_dim",
                        "depth",
                        "temporal_pool",
                        "spectral_bins",
                        "use_interactions",
                        "use_multiscale",
                        "attention_mix",
                    ],
                },
                "rationale": {"type": "string"},
                "logic_expression": {"type": "string"},
                "helper_code": {"type": "string"},
                "function_rewrite": {"type": "string"},
            },
            "required": [
                "model_name",
                "learning_rate",
                "model_params",
                "rationale",
                "logic_expression",
                "helper_code",
                "function_rewrite",
            ],
        }

        try:
            payload = planner._call_json_schema(
                name="autohar_model_repair",
                prompt=prompt,
                schema=schema,
                role="model_agent",
            )
            return ModelPlanHint(**payload)
        except Exception:
            return None

    def _repair_augmentation_proposal(
        self,
        context: dict[str, object],
        original: AugmentationProposal,
        strategy: str,
        attempt_num: int,
    ) -> AugmentationPlanHint | None:
        planner = self.planner
        if not hasattr(planner, "_call_json_schema"):
            return None

        prompt = (
            f"You are repairing a failed augmentation proposal (attempt {attempt_num}, strategy: {strategy}).\n\n"
            f"Error context:\n{json.dumps(context, indent=2)}\n\n"
            f"Original proposal: aug={original.augmentation}, strength={original.strength}\n\n"
            f"Return a simplified, safer augmentation proposal.\n"
            f"If the error was patch-related, fall back to 'none' or 'jitter' at 'weak'.\n"
        )
        schema = {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "augmentation": {"type": "string"},
                "strength": {"type": "string"},
                "rationale": {"type": "string"},
                "logic_expression": {"type": "string"},
                "helper_code": {"type": "string"},
                "function_rewrite": {"type": "string"},
            },
            "required": [
                "augmentation",
                "strength",
                "rationale",
                "logic_expression",
                "helper_code",
                "function_rewrite",
            ],
        }

        try:
            payload = planner._call_json_schema(
                name="autohar_augmentation_repair",
                prompt=prompt,
                schema=schema,
                role="augmentation_agent",
            )
            return AugmentationPlanHint(**payload)
        except Exception:
            return None

    def _repair_data_proposal(
        self,
        context: dict[str, object],
        original: DataProposal,
        strategy: str,
        attempt_num: int,
    ) -> DataPlanHint | None:
        planner = self.planner
        if not hasattr(planner, "_call_json_schema"):
            return None

        prompt = (
            f"You are repairing a failed data proposal (attempt {attempt_num}, strategy: {strategy}).\n\n"
            f"Error context:\n{json.dumps(context, indent=2)}\n\n"
            f"Original proposal: preprocessing={original.preprocessing}, window={original.window_size}\n\n"
            f"Return a simplified, safer data proposal.\n"
            f"Use standard window sizes: 64, 96, 128, 192, 256.\n"
        )
        schema = {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "preprocessing": {"type": "string"},
                "window_size": {"type": "integer"},
                "rationale": {"type": "string"},
                "logic_expression": {"type": "string"},
                "helper_code": {"type": "string"},
                "function_rewrite": {"type": "string"},
            },
            "required": [
                "preprocessing",
                "window_size",
                "rationale",
                "logic_expression",
                "helper_code",
                "function_rewrite",
            ],
        }

        try:
            payload = planner._call_json_schema(
                name="autohar_data_repair",
                prompt=prompt,
                schema=schema,
                role="data_agent",
            )
            return DataPlanHint(**payload)
        except Exception:
            return None

    def _heuristic_repair(
        self,
        proposals: list[ModelProposal | AugmentationProposal | DataProposal] | None,
        exc: Exception,
    ) -> list[ModelProposal | AugmentationProposal | DataProposal] | None:
        from autohar_researcher.heuristic_planner import HeuristicPlanner

        fallback = HeuristicPlanner()
        repaired = []

        for proposal in proposals or []:
            if isinstance(proposal, ModelProposal):
                hint = fallback._heuristic_model_hint(None, {}, 1)
                repaired.append(
                    ModelProposal(
                        model_name=hint.model_name,
                        learning_rate=hint.learning_rate,
                        model_params=dict(hint.model_params),
                        notes=f"[heuristic fallback after {_summarize_error(exc)}]",
                    )
                )
            elif isinstance(proposal, AugmentationProposal):
                hint = fallback._heuristic_augmentation_hint(None, {}, 1)
                repaired.append(
                    AugmentationProposal(
                        augmentation=hint.augmentation,
                        strength=hint.strength,
                        notes=f"[heuristic fallback after {_summarize_error(exc)}]",
                    )
                )
            else:
                hint = fallback._heuristic_data_hint(None, {}, 1)
                repaired.append(
                    DataProposal(
                        preprocessing=hint.preprocessing,
                        window_size=hint.window_size,
                        notes=f"[heuristic fallback after {_summarize_error(exc)}]",
                    )
                )

        return repaired if repaired else None

    def _write_repair_artifact(
        self,
        round_id: int,
        attempt_num: int,
        proposals: list[ModelProposal | AugmentationProposal | DataProposal],
    ) -> str:
        round_dir = ensure_round_dir(round_id)
        payload = []
        for p in proposals:
            if isinstance(p, ModelProposal):
                payload.append(
                    {
                        "agent": "model_agent",
                        "model_name": p.model_name,
                        "learning_rate": p.learning_rate,
                        "model_params": p.model_params,
                        "notes": p.notes,
                        "file_patches": [
                            {
                                "path": fp.file_path,
                                "mode": fp.patch_mode,
                                "rationale": fp.rationale,
                            }
                            for fp in p.file_patches
                        ],
                    }
                )
            elif isinstance(p, AugmentationProposal):
                payload.append(
                    {
                        "agent": "augmentation_agent",
                        "augmentation": p.augmentation,
                        "strength": p.strength,
                        "notes": p.notes,
                        "file_patches": [
                            {
                                "path": fp.file_path,
                                "mode": fp.patch_mode,
                                "rationale": fp.rationale,
                            }
                            for fp in p.file_patches
                        ],
                    }
                )
            else:
                payload.append(
                    {
                        "agent": "data_agent",
                        "preprocessing": p.preprocessing,
                        "window_size": p.window_size,
                        "notes": p.notes,
                        "file_patches": [
                            {
                                "path": fp.file_path,
                                "mode": fp.patch_mode,
                                "rationale": fp.rationale,
                            }
                            for fp in p.file_patches
                        ],
                    }
                )

        return write_json_artifact(
            round_dir / f"repair_attempt_{attempt_num}.json",
            {"attempt": attempt_num, "repaired_proposals": payload},
        )

    def _write_repair_record(self, round_id: int, record: RepairRecord) -> str:
        round_dir = ensure_round_dir(round_id)
        payload = {
            "round_id": record.round_id,
            "failure_type": record.failure_type,
            "error_traceback": record.error_traceback,
            "attempts": [
                {
                    "attempt": a.attempt,
                    "error": a.error_summary,
                    "strategy": a.repair_strategy,
                    "success": a.success,
                    "repair_artifact": a.repair_artifact,
                }
                for a in record.attempts
            ],
            "final_success": record.final_success,
            "fell_back_to_heuristic": record.fell_back_to_heuristic,
        }
        return write_json_artifact(round_dir / "repair_record.json", payload)
