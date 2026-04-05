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
    MODEL_FAMILIES = [
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
    AUGMENTATION_STRENGTHS = ["weak", "medium", "strong"]
    PREPROCESSING_MODES = [
        "raw",
        "normalized",
        "normalized+gravity_filter",
        "normalized+magnitude_channels",
    ]
    WINDOW_SIZES = [64, 96, 128, 192, 256]

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
            "combination_memory": context["combination_memory"],
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
            "exploration_pressure": self._exploration_pressure_summary(best_result, history, accepted_history),
            "combination_memory": self._combination_memory_summary(history, accepted_history),
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
            "Use focus values from {model, augmentation, data, combination, exploit, explore, stability} as compact labels, "
            "but rely on reasoning rather than menu-following.\n"
            "Use the combination_memory to reason explicitly about which single-agent, pairwise, or broader combinations are helping or conflicting.\n"
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
            "Common available model families include {deepconvlstm, deepconvlstmattention, hart, tinierhar, tinyhar, spectra, cnn_gru, cnn, tiny_transformer, gru}.\n"
            "Treat those families as stable anchors, not as the whole point of the search. "
            "Use them when they fit, but put most of your expertise into model-side helper logic, compatibility rules, and bounded rewrites inside the owned module.\n"
            "You may also set model_params such as hidden_dim, depth, temporal_pool, spectral_bins, use_interactions, use_multiscale, and attention_mix.\n"
            "Keep the rationale short and trend-aware.\n"
            "Use `logic_expression` only for trivial direct overrides.\n"
            "Prefer `helper_code` when the decision depends on baseline context, compatibility, guardrails, or conditional logic.\n"
            "Do not behave like a dropdown picker. "
            "If the important idea is a local model heuristic rather than a family switch, keep the family stable and express the novelty through `helper_code` or `function_rewrite`.\n"
            f"{rewrite_preference}"
            "When the main mutation is family-dependent parameter compatibility, prefer defining "
            "`model_params_override(...)` inside `function_rewrite` rather than only returning static model_params.\n"
            "In particular, use `model_params_override(...)` when hidden_dim, depth, temporal_pool, spectral_bins, "
            "use_interactions, use_multiscale, or attention_mix should depend on the current baseline context.\n"
            "For deeper code evolution, you may also provide `function_rewrite` as a full Python function definition for "
            "`bounded_learning_rate_override(base_experiment, target_name, baseline_name, baseline_lr, proposed_lr, config)`. "
            "Use this when you want to rewrite model learning-rate bounding logic itself inside the owned module. "
            "Return only a float learning rate or `None` to fall back.\n"
            "The same `function_rewrite` block may also define "
            "`model_params_override(base_experiment, target_name, proposed_params, config)` "
            "when you want to rewrite model-parameter compatibility logic itself inside the owned module. "
            "That function must return only a dict of model param overrides or `None` to fall back.\n"
            "Optionally provide a short Python expression in `logic_expression` that returns a dict of model-field overrides. "
            "The expression may reference `base_experiment` and `config`. Return only owned-section updates such as "
            "`{\"model\": {\"name\": ..., \"learning_rate\": ..., \"params\": {...}}}`. Use exact keys `name` and `learning_rate`. "
            "You may also return a lambda `(lambda base_experiment, config: {...})`. Use an empty string when unnecessary.\n"
            "Optionally provide `helper_code` containing small Python helper functions plus a top-level "
            "`llm_choose_patch(base_experiment, config)` function. That function must return only owned model updates. "
            "Keep helper_code short, self-contained, and limited to model-side reasoning. "
            "If you emit helper_code, it should usually be the primary mechanism and `logic_expression` can be empty.\n"
            "Prefer owned-module reasoning over menu hopping: use code when your idea is about architecture compatibility, family-dependent defaults, "
            "or training-shape heuristics rather than just swapping one named family for another.\n"
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
            "required": ["model_name", "learning_rate", "model_params", "rationale", "logic_expression", "helper_code", "function_rewrite"],
        }
        payload = self._call_json_schema(
            name="autohar_model_agent_plan",
            prompt=prompt,
            schema=schema,
        )
        payload = self._apply_model_exploration_pressure(context, payload)
        payload = self._prefer_model_param_rewrite(
            round_id=int(context["round_id"]),
            payload=payload,
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
            "Common stable augmentation families include {none, rotation, time_warp, jitter, mixup, scaling, drift, permutation, magnitude_warp}.\n"
            "Common strengths include {weak, medium, strong}.\n"
            "Treat those labels as anchors for the runtime, not as the main source of expertise. "
            "Put the interesting reasoning into augmentation-side parameter logic, compatibility rules, and bounded rewrites inside the owned module.\n"
            "Keep the rationale short and trend-aware.\n"
            "Use `logic_expression` only for trivial direct overrides.\n"
            "Prefer `helper_code` when the choice depends on baseline context, compatibility, guardrails, or conditional logic.\n"
            "Do not behave like a menu selector. "
            "If the important move is a context-sensitive perturbation policy, keep the family stable and express the novelty through `helper_code` or `function_rewrite`.\n"
            f"{rewrite_preference}"
            "When the main mutation is augmentation-family or strength-dependent parameter compatibility, prefer defining "
            "`augmentation_params_override(...)` inside `function_rewrite` rather than only relying on static defaults.\n"
            "Use `augmentation_params_override(...)` when rotation angle, jitter std, time-warp compression, scaling span, "
            "drift magnitude, permutation segments, magnitude-warp span, or mixup blend should depend on the current baseline context.\n"
            "For deeper code evolution, you may also provide `function_rewrite` as a full Python function definition for "
            "`compatibility_strength_override(base_experiment, target_name, requested_strength, config)`. "
            "Use this when you want to rewrite augmentation strength compatibility logic itself inside the owned module. "
            "Return only a valid strength string or `None` to fall back.\n"
            "The same `function_rewrite` block may also define "
            "`augmentation_params_override(base_experiment, target_name, target_strength, proposed_params, config)` "
            "when you want to rewrite augmentation-parameter compatibility logic itself inside the owned module. "
            "That function must return only a dict of augmentation param overrides or `None` to fall back.\n"
            "Optionally provide a short Python expression in `logic_expression` that returns a dict of augmentation-field overrides. "
            "The expression may reference `base_experiment` and `config`. Return only owned-section updates such as "
            "`{\"augmentation\": {\"name\": ..., \"strength\": ...}}`. Use exact keys `name` and `strength`. "
            "You may also return a lambda `(lambda base_experiment, config: {...})`. Use an empty string when unnecessary.\n"
            "Optionally provide `helper_code` containing small Python helper functions plus a top-level "
            "`llm_choose_patch(base_experiment, config)` function. That function must return only owned augmentation updates. "
            "Keep helper_code short, self-contained, and limited to augmentation-side reasoning. "
            "If you emit helper_code, it should usually be the primary mechanism and `logic_expression` can be empty.\n"
            "Prefer owned-module reasoning over family roulette: use code when your idea is about realism, robustness, parameter shaping, or compatibility "
            "with the current model/data context.\n"
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
        payload = self._apply_augmentation_exploration_pressure(context, payload)
        payload = self._prefer_augmentation_param_rewrite(
            round_id=int(context["round_id"]),
            payload=payload,
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
            "Common stable preprocessing regimes include {raw, normalized, normalized+gravity_filter, normalized+magnitude_channels}.\n"
            "Common stable window sizes include {64, 96, 128, 192, 256}.\n"
            "Treat those settings as anchors for the runtime, not as the whole search space. "
            "Put the interesting reasoning into data-side helper logic, compatibility rules, and bounded rewrites inside the owned module.\n"
            "Keep the rationale short and trend-aware.\n"
            "Use `logic_expression` only for trivial direct overrides.\n"
            "Prefer `helper_code` when the choice depends on baseline context, compatibility, guardrails, or conditional logic.\n"
            "Do not behave like a fixed menu chooser. "
            "If the real idea is a representation heuristic, keep the top-level setting stable and express the novelty through `helper_code` or `function_rewrite`.\n"
            f"{rewrite_preference}"
            "When the main mutation is preprocessing/window-dependent data parameter compatibility, prefer defining "
            "`data_params_override(...)` inside `function_rewrite` rather than only relying on static defaults.\n"
            "Use `data_params_override(...)` when channel layout, resampling mode, gravity scale, or magnitude toggles "
            "should depend on the current baseline context.\n"
            "For deeper code evolution, you may also provide `function_rewrite` as a full Python function definition for "
            "`choose_window_override(base_experiment, requested_window, config)`. "
            "Use this when you want to rewrite data window-selection logic itself inside the owned module. "
            "Return only an integer window size or `None` to fall back.\n"
            "The same `function_rewrite` block may also define "
            "`data_params_override(base_experiment, target_preprocessing, target_window, proposed_params, config)` "
            "when you want to rewrite data-parameter compatibility logic itself inside the owned module. "
            "That function must return only a dict of data param overrides or `None` to fall back.\n"
            "Optionally provide a short Python expression in `logic_expression` that returns a dict of data-field overrides. "
            "The expression may reference `base_experiment` and `config`. Return only owned-section updates such as "
            "`{\"data\": {\"preprocessing\": ..., \"window_size\": ...}}`. Use exact keys `preprocessing` and `window_size`. "
            "You may also return a lambda `(lambda base_experiment, config: {...})`. Use an empty string when unnecessary.\n"
            "Optionally provide `helper_code` containing small Python helper functions plus a top-level "
            "`llm_choose_patch(base_experiment, config)` function. That function must return only owned data updates. "
            "Keep helper_code short, self-contained, and limited to data-side reasoning. "
            "If you emit helper_code, it should usually be the primary mechanism and `logic_expression` can be empty.\n"
            "Prefer owned-module reasoning over cosmetic setting churn: use code when your idea is about representation quality, window compatibility, "
            "or preprocessing guardrails rather than just flipping to a neighboring preset.\n"
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
        payload = self._apply_data_exploration_pressure(context, payload)
        payload = self._prefer_data_param_rewrite(
            round_id=int(context["round_id"]),
            payload=payload,
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
        latest_round = max((result.round_id for result in history), default=0)
        last_accept_round = max((result.round_id for result in accepted_history), default=0)
        stagnation_rounds = max(0, latest_round - last_accept_round)
        return {
            "recent_best_by_agent": best_by_agent,
            "recent_avg_by_agent": avg_by_agent,
            "accepted_count": len(accepted_history),
            "recent_accept_plateau": plateau,
        }

    def _combination_memory_summary(
        self,
        history: list[CandidateResult],
        accepted_history: list[CandidateResult],
    ) -> dict[str, object]:
        labels = [
            "model_agent",
            "augmentation_agent",
            "data_agent",
            "model+augmentation",
            "model+data",
            "augmentation+data",
            "model+augmentation+data",
        ]
        by_label: dict[str, list[float]] = {label: [] for label in labels}
        accepted_by_label: dict[str, list[float]] = {label: [] for label in labels}
        synergy_records: list[dict[str, object]] = []
        round_groups: dict[int, list[CandidateResult]] = {}
        for result in history:
            round_groups.setdefault(result.round_id, []).append(result)
            if result.agent_name == "orchestrator":
                label = str(result.artifact_paths.get("selected_combination", "model+augmentation+data"))
            else:
                label = result.agent_name
            if label in by_label:
                by_label[label].append(result.val_score)
        for result in accepted_history:
            if result.agent_name == "orchestrator":
                label = str(result.artifact_paths.get("selected_combination", "model+augmentation+data"))
            else:
                label = result.agent_name
            if label in accepted_by_label:
                accepted_by_label[label].append(result.val_score)
        included_map = {
            "model+augmentation": ["model_agent", "augmentation_agent"],
            "model+data": ["model_agent", "data_agent"],
            "augmentation+data": ["augmentation_agent", "data_agent"],
            "model+augmentation+data": ["model_agent", "augmentation_agent", "data_agent"],
        }
        for round_id, round_results in round_groups.items():
            singles = {
                result.agent_name: result.val_score
                for result in round_results
                if result.agent_name in {"model_agent", "augmentation_agent", "data_agent"}
            }
            for result in round_results:
                if result.agent_name != "orchestrator":
                    continue
                label = str(result.artifact_paths.get("selected_combination", "model+augmentation+data"))
                included = included_map.get(label)
                if not included:
                    continue
                component_best = max((singles.get(name, float("-inf")) for name in included), default=float("-inf"))
                if component_best == float("-inf"):
                    continue
                synergy_records.append(
                    {
                        "round_id": round_id,
                        "label": label,
                        "score": round(result.val_score, 4),
                        "best_component_score": round(component_best, 4),
                        "synergy": round(result.val_score - component_best, 4),
                    }
                )
        recent_best = {
            label: round(max(scores), 4)
            for label, scores in by_label.items()
            if scores
        }
        recent_avg = {
            label: round(sum(scores) / len(scores), 4)
            for label, scores in by_label.items()
            if scores
        }
        accepted_best = {
            label: round(max(scores), 4)
            for label, scores in accepted_by_label.items()
            if scores
        }
        synergy_sorted = sorted(synergy_records, key=lambda item: item["synergy"], reverse=True)
        return {
            "recent_best_by_combination": recent_best,
            "recent_avg_by_combination": recent_avg,
            "accepted_best_by_combination": accepted_best,
            "compatible_patterns": synergy_sorted[:3],
            "conflicting_patterns": list(reversed(synergy_sorted[-3:])) if synergy_sorted else [],
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

    def _prefer_model_param_rewrite(
        self,
        round_id: int,
        payload: dict[str, object],
    ) -> dict[str, object]:
        if round_id <= 1:
            return payload
        if str(payload.get("function_rewrite", "")).strip():
            return payload
        model_params = payload.get("model_params")
        if not isinstance(model_params, dict) or not model_params:
            return payload
        serialized = json.dumps(model_params, sort_keys=True, indent=4)
        payload["function_rewrite"] = (
            "def model_params_override(base_experiment, target_name, proposed_params, config):\n"
            "    del base_experiment, target_name, config\n"
            f"    overrides = {serialized}\n"
            "    merged = dict(proposed_params)\n"
            "    merged.update(overrides)\n"
            "    return merged\n"
        )
        return payload

    def _prefer_augmentation_param_rewrite(
        self,
        round_id: int,
        payload: dict[str, object],
    ) -> dict[str, object]:
        if round_id <= 1:
            return payload
        if str(payload.get("function_rewrite", "")).strip():
            return payload
        augmentation = str(payload.get("augmentation", "none"))
        strength = str(payload.get("strength", "medium"))
        body = (
            "def augmentation_params_override(base_experiment, target_name, target_strength, proposed_params, config):\n"
            "    del base_experiment\n"
            "    if target_name != config.get('name', target_name):\n"
            "        return None\n"
            "    if target_strength != config.get('strength', target_strength):\n"
            "        return None\n"
            "    merged = dict(proposed_params)\n"
            "    merged.update(dict(config.get('params', {})))\n"
            "    return merged\n"
        )
        if augmentation == "none":
            return payload
        payload["function_rewrite"] = body
        return payload

    def _prefer_data_param_rewrite(
        self,
        round_id: int,
        payload: dict[str, object],
    ) -> dict[str, object]:
        if round_id <= 1:
            return payload
        if str(payload.get("function_rewrite", "")).strip():
            return payload
        preprocessing = str(payload.get("preprocessing", "raw"))
        window_size = int(payload.get("window_size", 128))
        body = (
            "def data_params_override(base_experiment, target_preprocessing, target_window, proposed_params, config):\n"
            "    del base_experiment\n"
            "    if target_preprocessing != config.get('preprocessing', target_preprocessing):\n"
            "        return None\n"
            "    if int(target_window) != int(config.get('window_size', target_window)):\n"
            "        return None\n"
            "    merged = dict(proposed_params)\n"
            "    merged.update(dict(config.get('params', {})))\n"
            "    return merged\n"
        )
        if preprocessing == "raw" and window_size == 128:
            return payload
        payload["function_rewrite"] = body
        return payload

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

    def _exploration_pressure_summary(
        self,
        best_result: CandidateResult | None,
        history: list[CandidateResult],
        accepted_history: list[CandidateResult],
    ) -> dict[str, object]:
        def repeated_count(values: list[object]) -> int:
            if not values:
                return 0
            last = values[-1]
            count = 0
            for value in reversed(values):
                if value == last:
                    count += 1
                else:
                    break
            return count

        plateau = False
        if len(accepted_history) >= 3:
            recent_accepts = [result.val_score for result in accepted_history[-3:]]
            plateau = max(recent_accepts) - min(recent_accepts) < 0.01

        model_values = [result.config.model_name for result in history if result.agent_name == "model_agent"]
        augmentation_values = [
            (result.config.augmentation, result.config.augmentation_strength)
            for result in history
            if result.agent_name == "augmentation_agent"
        ]
        data_values = [
            (result.config.preprocessing, int(result.config.window_size))
            for result in history
            if result.agent_name == "data_agent"
        ]
        baseline_config = None if best_result is None else asdict(best_result.config)
        return {
            "plateau": plateau,
            "stagnation_rounds": stagnation_rounds,
            "baseline_config": baseline_config,
            "model_repeat_streak": repeated_count(model_values),
            "augmentation_repeat_streak": repeated_count(augmentation_values),
            "data_repeat_streak": repeated_count(data_values),
        }

    def _apply_model_exploration_pressure(
        self,
        context: dict[str, object],
        payload: dict[str, object],
    ) -> dict[str, object]:
        pressure = dict(context.get("exploration_pressure", {}))
        baseline = dict(pressure.get("baseline_config") or {})
        baseline_model = str(baseline.get("model_name", ""))
        repeat_streak = int(pressure.get("model_repeat_streak", 0))
        plateau = bool(pressure.get("plateau", False))
        stagnation_rounds = int(pressure.get("stagnation_rounds", 0))
        current_model = str(payload.get("model_name", ""))
        if current_model != baseline_model:
            return payload
        if stagnation_rounds < 5 and not plateau and repeat_streak < 3:
            return payload
        recent_models = [
            str(item["config"]["model_name"])
            for item in context["agent_memories"]["model_agent"].get("recent_trials", [])
            if isinstance(item, dict)
        ]
        candidate_order = self.MODEL_FAMILIES
        if stagnation_rounds >= 5:
            candidate_order = [
                candidate
                for candidate in self.MODEL_FAMILIES
                if candidate not in {baseline_model, "deepconvlstm", "deepconvlstmattention"}
            ] + self.MODEL_FAMILIES
        for candidate in candidate_order:
            if candidate != baseline_model and candidate not in recent_models[-3:]:
                payload["model_name"] = candidate
                payload["rationale"] = (
                    f"{payload.get('rationale', '').strip()} "
                    f"[exploration pressure: diversify away from repeated/stagnant baseline family {baseline_model} -> {candidate}]"
                ).strip()
                break
        return payload

    def _apply_augmentation_exploration_pressure(
        self,
        context: dict[str, object],
        payload: dict[str, object],
    ) -> dict[str, object]:
        pressure = dict(context.get("exploration_pressure", {}))
        baseline = dict(pressure.get("baseline_config") or {})
        baseline_aug = str(baseline.get("augmentation", ""))
        baseline_strength = str(baseline.get("augmentation_strength", "medium"))
        repeat_streak = int(pressure.get("augmentation_repeat_streak", 0))
        plateau = bool(pressure.get("plateau", False))
        stagnation_rounds = int(pressure.get("stagnation_rounds", 0))
        current = (str(payload.get("augmentation", "")), str(payload.get("strength", "")))
        if current != (baseline_aug, baseline_strength):
            return payload
        if stagnation_rounds < 5 and not plateau and repeat_streak < 3:
            return payload
        recent_values = [
            (
                str(item["config"]["augmentation"]),
                str(item["config"]["augmentation_strength"]),
            )
            for item in context["agent_memories"]["augmentation_agent"].get("recent_trials", [])
            if isinstance(item, dict)
        ]
        augmentation_order = self.AUGMENTATIONS
        if stagnation_rounds >= 5:
            augmentation_order = [aug for aug in self.AUGMENTATIONS if aug not in {baseline_aug, "none"}] + self.AUGMENTATIONS
        for augmentation in augmentation_order:
            for strength in self.AUGMENTATION_STRENGTHS:
                candidate = (augmentation, strength)
                if candidate != (baseline_aug, baseline_strength) and candidate not in recent_values[-3:]:
                    payload["augmentation"] = augmentation
                    payload["strength"] = strength
                    payload["rationale"] = (
                        f"{payload.get('rationale', '').strip()} "
                        f"[exploration pressure: diversify away from repeated/stagnant baseline augmentation {baseline_aug}:{baseline_strength} -> {augmentation}:{strength}]"
                    ).strip()
                    return payload
        return payload

    def _apply_data_exploration_pressure(
        self,
        context: dict[str, object],
        payload: dict[str, object],
    ) -> dict[str, object]:
        pressure = dict(context.get("exploration_pressure", {}))
        baseline = dict(pressure.get("baseline_config") or {})
        baseline_pre = str(baseline.get("preprocessing", ""))
        baseline_window = int(baseline.get("window_size", 128)) if baseline else 128
        repeat_streak = int(pressure.get("data_repeat_streak", 0))
        plateau = bool(pressure.get("plateau", False))
        stagnation_rounds = int(pressure.get("stagnation_rounds", 0))
        current = (str(payload.get("preprocessing", "")), int(payload.get("window_size", 128)))
        if current != (baseline_pre, baseline_window):
            return payload
        if stagnation_rounds < 5 and not plateau and repeat_streak < 3:
            return payload
        recent_values = [
            (
                str(item["config"]["preprocessing"]),
                int(item["config"]["window_size"]),
            )
            for item in context["agent_memories"]["data_agent"].get("recent_trials", [])
            if isinstance(item, dict)
        ]
        preprocessing_order = self.PREPROCESSING_MODES
        window_order = self.WINDOW_SIZES
        if stagnation_rounds >= 5:
            preprocessing_order = [mode for mode in self.PREPROCESSING_MODES if mode != baseline_pre] + self.PREPROCESSING_MODES
            window_order = [size for size in self.WINDOW_SIZES if size != baseline_window] + self.WINDOW_SIZES
        for preprocessing in preprocessing_order:
            for window_size in window_order:
                candidate = (preprocessing, window_size)
                if candidate != (baseline_pre, baseline_window) and candidate not in recent_values[-3:]:
                    payload["preprocessing"] = preprocessing
                    payload["window_size"] = window_size
                    payload["rationale"] = (
                        f"{payload.get('rationale', '').strip()} "
                        f"[exploration pressure: diversify away from repeated/stagnant baseline data setting {baseline_pre}:{baseline_window} -> {preprocessing}:{window_size}]"
                    ).strip()
                    return payload
        return payload


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
