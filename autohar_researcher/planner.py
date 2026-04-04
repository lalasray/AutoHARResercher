from __future__ import annotations

import json
import os
from dataclasses import asdict
from pathlib import Path

try:
    from openai import OpenAI
except Exception:  # pragma: no cover - optional dependency
    OpenAI = None

from autohar_researcher.types import (
    AugmentationPlanHint,
    CandidateResult,
    DataPlanHint,
    ModelPlanHint,
)


def load_local_env(path: Path | None = None) -> None:
    env_path = path or (Path.cwd() / ".autohar.env")
    if not env_path.exists():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


class OpenAIPlanner:
    name = "openai_responses_multiagent"

    def __init__(self, model: str | None = None) -> None:
        if OpenAI is None:
            raise RuntimeError("openai package is not installed")
        self.model = model or os.getenv("AUTOHAR_LLM_MODEL", "gpt-5-mini")
        self.client = OpenAI()
        self.guides = self._load_guides()

    def plan_round(
        self,
        round_id: int,
        best_result: CandidateResult | None,
        history: list[CandidateResult],
        accepted_history: list[CandidateResult],
    ) -> dict[str, object]:
        context = self._shared_context(
            round_id=round_id,
            best_result=best_result,
            history=history,
            accepted_history=accepted_history,
        )
        orchestrator_plan = self._plan_orchestrator(context)
        hints: dict[str, object] = {
            "model_agent": self._plan_model_agent(context, orchestrator_plan),
            "augmentation_agent": self._plan_augmentation_agent(context, orchestrator_plan),
            "data_agent": self._plan_data_agent(context, orchestrator_plan),
        }
        agent_memories = context["agent_memories"]
        return {
            "planner": self.name,
            "model": self.model,
            "summary": orchestrator_plan["summary"],
            "hints": hints,
            "orchestrator_plan": orchestrator_plan,
            "agent_memories": agent_memories,
        }

    def _shared_context(
        self,
        round_id: int,
        best_result: CandidateResult | None,
        history: list[CandidateResult],
        accepted_history: list[CandidateResult],
    ) -> dict[str, object]:
        trend_summary = self._trend_summary(history, accepted_history)
        recent_trials = [
            {
                "round_id": result.round_id,
                "agent_name": result.agent_name,
                "score": round(result.val_score, 4),
                "config": asdict(result.config),
                "summary": result.summary,
            }
            for result in history[-12:]
        ]
        accepted = [
            {
                "round_id": result.round_id,
                "agent_name": result.agent_name,
                "score": round(result.val_score, 4),
                "config": asdict(result.config),
                "summary": result.summary,
            }
            for result in accepted_history[-6:]
        ]
        best_payload = None
        if best_result is not None:
            best_payload = {
                "round_id": best_result.round_id,
                "agent_name": best_result.agent_name,
                "branch_name": best_result.branch_name,
                "score": round(best_result.val_score, 4),
                "config": asdict(best_result.config),
                "summary": best_result.summary,
            }
        return {
            "round_id": round_id,
            "best_result": best_payload,
            "trend_summary": trend_summary,
            "recent_trials": recent_trials,
            "accepted_history": accepted,
            "agent_memories": {
                "model_agent": self._agent_memory_summary("model_agent", history, accepted_history),
                "augmentation_agent": self._agent_memory_summary("augmentation_agent", history, accepted_history),
                "data_agent": self._agent_memory_summary("data_agent", history, accepted_history),
            },
        }

    def _plan_orchestrator(self, context: dict[str, object]) -> dict[str, object]:
        prompt = (
            "You are the orchestrator of an IMU HAR auto-research system.\n"
            "Your job is to decide the round-level search strategy before the specialist agents act.\n"
            "You are not choosing the final config directly. You are setting priorities and guardrails for the three specialist agents.\n\n"
            f"Orchestrator guide:\n{self.guides['orchestrator']}\n\n"
            f"Round context:\n{json.dumps(context, indent=2)}\n\n"
            "Return a concise JSON strategy for this round.\n"
            "The accepted baseline should only be replaced if a candidate scores better.\n"
            "Use valid focus values from {model, augmentation, data, combination, exploit, explore, stability}.\n"
        )
        schema = {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "summary": {"type": "string"},
                "priority_order": {
                    "type": "array",
                    "items": {"type": "string"},
                    "minItems": 3,
                    "maxItems": 3,
                },
                "exploration_mode": {"type": "string"},
                "combination_bias": {"type": "string"},
                "risk_note": {"type": "string"},
            },
            "required": [
                "summary",
                "priority_order",
                "exploration_mode",
                "combination_bias",
                "risk_note",
            ],
        }
        return self._call_json_schema(
            name="autohar_orchestrator_plan",
            prompt=prompt,
            schema=schema,
        )

    def _plan_model_agent(
        self,
        context: dict[str, object],
        orchestrator_plan: dict[str, object],
    ) -> ModelPlanHint:
        rewrite_preference = self._rewrite_preference_note(int(context["round_id"]))
        prompt = (
            "You are ModelAgent in an IMU HAR auto-research system.\n"
            "You only control model_name, learning_rate, and model-side params.\n"
            "Propose one model-side mutation for this round.\n"
            "Do not change augmentation or data settings.\n\n"
            f"ModelAgent guide:\n{self.guides['model_agent']}\n\n"
            f"Orchestrator strategy:\n{json.dumps(orchestrator_plan, indent=2)}\n\n"
            f"ModelAgent local memory:\n{json.dumps(context['agent_memories']['model_agent'], indent=2)}\n\n"
            f"Round context:\n{json.dumps(context, indent=2)}\n\n"
            "Use valid models from {deepconvlstm, hart, tinierhar, tinyhar, spectra, cnn_gru, cnn, tiny_transformer, gru}.\n"
            "You may also set model_params such as hidden_dim, depth, temporal_pool, spectral_bins, use_interactions, and use_multiscale.\n"
            "Keep the rationale short and trend-aware.\n"
            "Use `logic_expression` only for trivial direct overrides.\n"
            "Prefer `helper_code` when the decision depends on baseline context, compatibility, guardrails, or conditional logic.\n"
            f"{rewrite_preference}"
            "For deeper code evolution, you may also provide `function_rewrite` as a full Python function definition for "
            "`bounded_learning_rate_override(base_experiment, target_name, baseline_name, baseline_lr, proposed_lr, config)`. "
            "Use this when you want to rewrite model learning-rate bounding logic itself inside the owned module. "
            "Return only a float learning rate or `None` to fall back.\n"
            "Optionally provide a short Python expression in `logic_expression` that returns a dict of model-field overrides. "
            "The expression may reference `base_experiment` and `config`. Return only owned-section updates such as "
            "`{\"model\": {\"name\": ..., \"learning_rate\": ..., \"params\": {...}}}`. Use exact keys `name` and `learning_rate`. "
            "You may also return a lambda `(lambda base_experiment, config: {...})`. Use an empty string when unnecessary.\n"
            "Optionally provide `helper_code` containing small Python helper functions plus a top-level "
            "`llm_choose_patch(base_experiment, config)` function. That function must return only owned model updates. "
            "Keep helper_code short, self-contained, and limited to model-side reasoning. "
            "If you emit helper_code, it should usually be the primary mechanism and `logic_expression` can be empty.\n"
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
                    },
                    "required": [
                        "hidden_dim",
                        "depth",
                        "temporal_pool",
                        "spectral_bins",
                        "use_interactions",
                        "use_multiscale",
                    ],
                },
                "rationale": {"type": "string"},
                "logic_expression": {"type": "string"},
                "helper_code": {"type": "string"},
                "function_rewrite": {"type": "string"},
            },
            "required": ["model_name", "learning_rate", "model_params", "rationale", "logic_expression", "helper_code", "function_rewrite"],
        }
        payload = self._call_json_schema(
            name="autohar_model_agent_plan",
            prompt=prompt,
            schema=schema,
        )
        return ModelPlanHint(**payload)

    def _plan_augmentation_agent(
        self,
        context: dict[str, object],
        orchestrator_plan: dict[str, object],
    ) -> AugmentationPlanHint:
        rewrite_preference = self._rewrite_preference_note(int(context["round_id"]))
        prompt = (
            "You are AugmentationAgent in an IMU HAR auto-research system.\n"
            "You only control augmentation and strength.\n"
            "Propose one augmentation-side mutation for this round.\n"
            "Do not change model or data settings.\n\n"
            f"AugmentationAgent guide:\n{self.guides['augmentation_agent']}\n\n"
            f"Orchestrator strategy:\n{json.dumps(orchestrator_plan, indent=2)}\n\n"
            f"AugmentationAgent local memory:\n{json.dumps(context['agent_memories']['augmentation_agent'], indent=2)}\n\n"
            f"Round context:\n{json.dumps(context, indent=2)}\n\n"
            "Use valid augmentations from {none, rotation, time_warp, jitter, mixup, scaling, drift, permutation, magnitude_warp}.\n"
            "Use valid strengths from {weak, medium, strong}.\n"
            "Keep the rationale short and trend-aware.\n"
            "Use `logic_expression` only for trivial direct overrides.\n"
            "Prefer `helper_code` when the choice depends on baseline context, compatibility, guardrails, or conditional logic.\n"
            f"{rewrite_preference}"
            "For deeper code evolution, you may also provide `function_rewrite` as a full Python function definition for "
            "`compatibility_strength_override(base_experiment, target_name, requested_strength, config)`. "
            "Use this when you want to rewrite augmentation strength compatibility logic itself inside the owned module. "
            "Return only a valid strength string or `None` to fall back.\n"
            "Optionally provide a short Python expression in `logic_expression` that returns a dict of augmentation-field overrides. "
            "The expression may reference `base_experiment` and `config`. Return only owned-section updates such as "
            "`{\"augmentation\": {\"name\": ..., \"strength\": ...}}`. Use exact keys `name` and `strength`. "
            "You may also return a lambda `(lambda base_experiment, config: {...})`. Use an empty string when unnecessary.\n"
            "Optionally provide `helper_code` containing small Python helper functions plus a top-level "
            "`llm_choose_patch(base_experiment, config)` function. That function must return only owned augmentation updates. "
            "Keep helper_code short, self-contained, and limited to augmentation-side reasoning. "
            "If you emit helper_code, it should usually be the primary mechanism and `logic_expression` can be empty.\n"
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
            "required": ["augmentation", "strength", "rationale", "logic_expression", "helper_code", "function_rewrite"],
        }
        payload = self._call_json_schema(
            name="autohar_augmentation_agent_plan",
            prompt=prompt,
            schema=schema,
        )
        return AugmentationPlanHint(**payload)

    def _plan_data_agent(
        self,
        context: dict[str, object],
        orchestrator_plan: dict[str, object],
    ) -> DataPlanHint:
        rewrite_preference = self._rewrite_preference_note(int(context["round_id"]))
        prompt = (
            "You are DataAgent in an IMU HAR auto-research system.\n"
            "You only control preprocessing and window_size.\n"
            "Propose one data-side mutation for this round.\n"
            "Do not change model or augmentation settings.\n\n"
            f"DataAgent guide:\n{self.guides['data_agent']}\n\n"
            f"Orchestrator strategy:\n{json.dumps(orchestrator_plan, indent=2)}\n\n"
            f"DataAgent local memory:\n{json.dumps(context['agent_memories']['data_agent'], indent=2)}\n\n"
            f"Round context:\n{json.dumps(context, indent=2)}\n\n"
            "Use valid preprocessing from {raw, normalized, normalized+gravity_filter, normalized+magnitude_channels}.\n"
            "Use valid window sizes from {64, 96, 128, 192, 256}.\n"
            "Keep the rationale short and trend-aware.\n"
            "Use `logic_expression` only for trivial direct overrides.\n"
            "Prefer `helper_code` when the choice depends on baseline context, compatibility, guardrails, or conditional logic.\n"
            f"{rewrite_preference}"
            "For deeper code evolution, you may also provide `function_rewrite` as a full Python function definition for "
            "`choose_window_override(base_experiment, requested_window, config)`. "
            "Use this when you want to rewrite data window-selection logic itself inside the owned module. "
            "Return only an integer window size or `None` to fall back.\n"
            "Optionally provide a short Python expression in `logic_expression` that returns a dict of data-field overrides. "
            "The expression may reference `base_experiment` and `config`. Return only owned-section updates such as "
            "`{\"data\": {\"preprocessing\": ..., \"window_size\": ...}}`. Use exact keys `preprocessing` and `window_size`. "
            "You may also return a lambda `(lambda base_experiment, config: {...})`. Use an empty string when unnecessary.\n"
            "Optionally provide `helper_code` containing small Python helper functions plus a top-level "
            "`llm_choose_patch(base_experiment, config)` function. That function must return only owned data updates. "
            "Keep helper_code short, self-contained, and limited to data-side reasoning. "
            "If you emit helper_code, it should usually be the primary mechanism and `logic_expression` can be empty.\n"
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
            "required": ["preprocessing", "window_size", "rationale", "logic_expression", "helper_code", "function_rewrite"],
        }
        payload = self._call_json_schema(
            name="autohar_data_agent_plan",
            prompt=prompt,
            schema=schema,
        )
        return DataPlanHint(**payload)

    def _call_json_schema(
        self,
        name: str,
        prompt: str,
        schema: dict[str, object],
    ) -> dict[str, object]:
        response = self.client.responses.create(
            model=self.model,
            input=prompt,
            text={
                "format": {
                    "type": "json_schema",
                    "name": name,
                    "schema": schema,
                    "strict": True,
                }
            },
        )
        return json.loads(response.output_text)

    def _load_guides(self) -> dict[str, str]:
        guide_root = Path.cwd() / "agent_guides"
        return {
            "orchestrator": (guide_root / "orchestrator.md").read_text(encoding="utf-8"),
            "model_agent": (guide_root / "model_agent.md").read_text(encoding="utf-8"),
            "augmentation_agent": (guide_root / "augmentation_agent.md").read_text(encoding="utf-8"),
            "data_agent": (guide_root / "data_agent.md").read_text(encoding="utf-8"),
        }

    def _trend_summary(
        self,
        history: list[CandidateResult],
        accepted_history: list[CandidateResult],
    ) -> dict[str, object]:
        recent = history[-20:]
        best_by_agent: dict[str, float] = {}
        avg_by_agent: dict[str, float] = {}
        if recent:
            agent_names = sorted({result.agent_name for result in recent})
            for agent_name in agent_names:
                scores = [result.val_score for result in recent if result.agent_name == agent_name]
                best_by_agent[agent_name] = round(max(scores), 4)
                avg_by_agent[agent_name] = round(sum(scores) / len(scores), 4)
        plateau = False
        if len(accepted_history) >= 3:
            recent_accepts = [result.val_score for result in accepted_history[-3:]]
            plateau = max(recent_accepts) - min(recent_accepts) < 0.01
        return {
            "recent_best_by_agent": best_by_agent,
            "recent_avg_by_agent": avg_by_agent,
            "accepted_count": len(accepted_history),
            "recent_accept_plateau": plateau,
        }

    def _rewrite_preference_note(self, round_id: int) -> str:
        if round_id <= 1:
            return (
                "In round 1, bounded function rewrites are optional; prefer them only when they add real value.\n"
            )
        return (
            "For round 2 and later, bounded `function_rewrite` is the preferred specialist path whenever the mutation depends on local reusable logic. "
            "Leave it empty only if the proposal is truly a trivial direct override.\n"
        )

    def _agent_memory_summary(
        self,
        agent_name: str,
        history: list[CandidateResult],
        accepted_history: list[CandidateResult],
    ) -> dict[str, object]:
        scoped_history = [result for result in history if result.agent_name == agent_name]
        accepted_scoped = [result for result in accepted_history if result.agent_name == agent_name]
        recent_trials = [
            {
                "round_id": result.round_id,
                "score": round(result.val_score, 4),
                "config": asdict(result.config),
                "summary": result.summary,
            }
            for result in scoped_history[-6:]
        ]
        accepted = [
            {
                "round_id": result.round_id,
                "score": round(result.val_score, 4),
                "config": asdict(result.config),
                "summary": result.summary,
            }
            for result in accepted_scoped[-4:]
        ]
        best_trial = None
        if scoped_history:
            best = max(scoped_history, key=lambda result: result.val_score)
            best_trial = {
                "round_id": best.round_id,
                "score": round(best.val_score, 4),
                "config": asdict(best.config),
                "summary": best.summary,
            }
        return {
            "agent_name": agent_name,
            "recent_trials": recent_trials,
            "accepted_history": accepted,
            "best_trial": best_trial,
            "trial_count": len(scoped_history),
            "accepted_count": len(accepted_scoped),
        }


def build_planner() -> OpenAIPlanner:
    load_local_env()
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError(
            "OPENAI_API_KEY is required for planning. "
            "Put it in .autohar.env or export it in your shell."
        )
    if OpenAI is None:
        raise RuntimeError(
            "The openai Python package is required for planning. "
            "Install project dependencies first."
        )
    return OpenAIPlanner()
