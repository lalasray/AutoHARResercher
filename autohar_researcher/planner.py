from __future__ import annotations

import json
import os
import re
import subprocess
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING

try:
    from openai import OpenAI
except Exception:  # pragma: no cover - optional dependency
    OpenAI = None

from autohar_researcher.types import (
    AugmentationPlanHint,
    CandidateResult,
    DataPlanHint,
    FilePatch,
    ModelPlanHint,
    ProposalJudgment,
)
from autohar_researcher.datasets import UCIHARDataset

if TYPE_CHECKING:
    from autohar_researcher.heuristic_planner import HeuristicPlanner


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
    ROLE_KEYS = {
        "orchestrator": "ORCHESTRATOR",
        "model_agent": "MODEL_AGENT",
        "augmentation_agent": "AUGMENTATION_AGENT",
        "data_agent": "DATA_AGENT",
        "judge": "JUDGE",
    }
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
        "custom",
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
        "custom",
    ]
    AUGMENTATION_STRENGTHS = ["weak", "medium", "strong"]
    PREPROCESSING_MODES = [
        "raw",
        "normalized",
        "normalized+gravity_filter",
        "normalized+magnitude_channels",
        "custom",
    ]
    WINDOW_SIZES = [64, 96, 128, 192, 256]

    def __init__(
        self,
        model: str | None = None,
        *,
        base_url: str | None = None,
        api_key: str | None = None,
        planner_name: str | None = None,
        api_mode: str = "responses",
        mission_brief: str | None = None,
        mission_source: str | None = None,
    ) -> None:
        if api_mode == "opencode":
            self.model = model or os.getenv("AUTOHAR_LLM_MODEL", "")
        else:
            self.model = model or os.getenv("AUTOHAR_LLM_MODEL", "gpt-5-mini")
        self.base_url = base_url
        self.api_mode = api_mode
        self.name = planner_name or self.name
        self.default_api_key = api_key
        self._client_cache: dict[tuple[str, str, str, str], OpenAI] = {}
        self._opencode_model_cache: list[str] | None = None
        self.guides = self._load_guides()
        self.mission_brief = (mission_brief or "").strip()
        self.mission_source = mission_source

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
        prompt_context = self._prompt_context(context)
        orchestrator_plan = self._plan_orchestrator(prompt_context)
        orchestrator_plan = self._apply_stagnation_overrides_to_orchestrator_plan(
            context=context,
            orchestrator_plan=orchestrator_plan,
        )
        hints: dict[str, object] = {
            "model_agent": self._plan_model_agent(prompt_context, orchestrator_plan),
            "augmentation_agent": self._plan_augmentation_agent(
                prompt_context, orchestrator_plan
            ),
            "data_agent": self._plan_data_agent(prompt_context, orchestrator_plan),
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

    def judge_proposals(
        self,
        round_id: int,
        best_result: CandidateResult | None,
        history: list[CandidateResult],
        accepted_history: list[CandidateResult],
        orchestrator_plan: dict[str, object],
        proposals: dict[str, object],
    ) -> dict[str, ProposalJudgment]:
        context = self._shared_context(
            round_id=round_id,
            best_result=best_result,
            history=history,
            accepted_history=accepted_history,
        )
        prompt_context = self._prompt_context(context)
        proposal_payload = {
            agent_name: self._proposal_payload(proposal)
            for agent_name, proposal in proposals.items()
        }
        prompt = (
            "You are an IMU HAR proposal judge.\n"
            "You are not the final evaluator. The harness is the final truth.\n"
            "Your role is to judge whether each specialist proposal is coherent, novel enough, and compatible enough to be worth using in orchestrator composition this round.\n\n"
            f"Round context:\n{json.dumps(prompt_context, indent=2)}\n\n"
            f"Orchestrator strategy:\n{json.dumps(orchestrator_plan, indent=2)}\n\n"
            f"Specialist proposals:\n{json.dumps(proposal_payload, indent=2)}\n\n"
            "For each agent, return:\n"
            "- mutation_size: one of {small, medium, large}\n"
            "- novelty: one of {low, medium, high}\n"
            "- plausibility: one of {low, medium, high}\n"
            "- compatibility: one of {low, medium, high}\n"
            "- evaluate_now: true if this proposal is worth using in orchestrator composition decisions this round\n"
            "- rationale: short explanation\n"
            "Use evaluate_now=false only for proposals that look incoherent, invalid, self-contradictory, or clearly unsafe for the current baseline. "
            "Do not over-police moderately bold but plausible proposals.\n"
        )
        schema = {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "model_agent": self._judgment_schema(),
                "augmentation_agent": self._judgment_schema(),
                "data_agent": self._judgment_schema(),
            },
            "required": ["model_agent", "augmentation_agent", "data_agent"],
        }
        payload = self._call_json_schema(
            name="autohar_proposal_judgments",
            prompt=prompt,
            schema=schema,
            role="judge",
        )
        payload = self._relax_judgments(context=context, payload=payload)
        return {
            agent_name: ProposalJudgment(
                agent_name=agent_name, **dict(judgment_payload)
            )
            for agent_name, judgment_payload in payload.items()
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
            "mission_brief": self._mission_payload(),
            "dataset_metadata": self._dataset_metadata(best_result),
            "trend_summary": trend_summary,
            "recent_trials": recent_trials,
            "accepted_history": accepted,
            "exploration_pressure": self._exploration_pressure_summary(
                best_result, history, accepted_history
            ),
            "combination_memory": self._combination_memory_summary(
                history, accepted_history
            ),
            "agent_memories": {
                "model_agent": self._agent_memory_summary(
                    "model_agent", history, accepted_history
                ),
                "augmentation_agent": self._agent_memory_summary(
                    "augmentation_agent", history, accepted_history
                ),
                "data_agent": self._agent_memory_summary(
                    "data_agent", history, accepted_history
                ),
            },
        }

    def _prompt_context(self, context: dict[str, object]) -> dict[str, object]:
        compact_mode = (
            os.getenv("AUTOHAR_COMPACT_PROMPT_CONTEXT", "false").strip().lower()
            == "true"
        )
        if not compact_mode:
            return context
        best_result = context.get("best_result")
        return {
            "round_id": context.get("round_id"),
            "best_result": self._compact_result(best_result),
            "mission_brief": self._compact_mission_payload(
                context.get("mission_brief")
            ),
            "dataset_metadata": self._compact_dataset_metadata(
                dict(context.get("dataset_metadata", {}))
            ),
            "trend_summary": context.get("trend_summary", {}),
            "exploration_pressure": context.get("exploration_pressure", {}),
            "combination_memory": self._compact_combination_memory(
                dict(context.get("combination_memory", {}))
            ),
            "recent_trials": [
                self._compact_result(item)
                for item in list(context.get("recent_trials", []))[-6:]
            ],
            "accepted_history": [
                self._compact_result(item)
                for item in list(context.get("accepted_history", []))[-3:]
            ],
            "agent_memories": self._compact_agent_memories(
                dict(context.get("agent_memories", {}))
            ),
        }

    def _role_backend(self, role: str) -> str:
        role_key = self.ROLE_KEYS.get(role, role.upper())
        return os.getenv(
            f"AUTOHAR_{role_key}_BACKEND",
            os.getenv("AUTOHAR_LLM_BACKEND", "openai"),
        )

    def _role_api_mode(self, role: str, backend: str) -> str:
        role_key = self.ROLE_KEYS.get(role, role.upper())
        if backend == "lmstudio":
            return os.getenv(
                f"AUTOHAR_LMSTUDIO_{role_key}_API_MODE",
                os.getenv(f"AUTOHAR_{role_key}_API_MODE", "chat"),
            )
        return os.getenv(
            f"AUTOHAR_OPENAI_{role_key}_API_MODE",
            os.getenv(f"AUTOHAR_{role_key}_API_MODE", self.api_mode),
        )

    def _compact_result(self, item: object) -> dict[str, object] | None:
        if not isinstance(item, dict):
            return None
        config = dict(item.get("config", {})) if isinstance(item.get("config"), dict) else {}
        config.pop("notes", None)
        for key in ("model_params", "data_params", "augmentation_params"):
            if isinstance(config.get(key), dict):
                config[key] = self._compact_params(dict(config[key]))
        return {
            "round_id": item.get("round_id"),
            "agent_name": item.get("agent_name"),
            "score": item.get("score"),
            "config": config,
        }

    def _mission_payload(self) -> dict[str, object] | None:
        if not self.mission_brief:
            return None
        return {
            "source": self.mission_source,
            "content": self.mission_brief,
        }

    def _compact_mission_payload(
        self, payload: object, max_chars: int = 2400
    ) -> dict[str, object] | None:
        if not isinstance(payload, dict):
            return None
        content = str(payload.get("content", "")).strip()
        if not content:
            return None
        if len(content) > max_chars:
            content = content[: max_chars - 15].rstrip() + "\n...[truncated]"
        return {
            "source": payload.get("source"),
            "content": content,
        }

    def _compact_params(self, params: dict[str, object]) -> dict[str, object]:
        compact = dict(params)
        if "custom_code" in compact:
            compact["custom_code"] = "[omitted]"
        return compact

    def _compact_dataset_metadata(self, metadata: dict[str, object]) -> dict[str, object]:
        return {
            key: metadata.get(key)
            for key in (
                "dataset_name",
                "activity_labels",
                "sensor_count",
                "sensor_modalities",
                "sensor_placement",
                "sampling_rate_hz",
                "window_length_steps",
            )
            if key in metadata
        }

    def _compact_combination_memory(
        self, memory: dict[str, object]
    ) -> dict[str, object]:
        compact: dict[str, object] = {}
        for key in (
            "recent_best_by_combination",
            "accepted_best_by_combination",
            "compatible_patterns",
            "conflicting_patterns",
        ):
            value = memory.get(key)
            if isinstance(value, list):
                compact[key] = value[:4]
            elif isinstance(value, dict):
                compact[key] = value
        return compact

    def _compact_agent_memories(
        self, memories: dict[str, object]
    ) -> dict[str, object]:
        compact: dict[str, object] = {}
        for agent_name, memory in memories.items():
            if not isinstance(memory, dict):
                continue
            compact[agent_name] = {
                "best_score": memory.get("best_score"),
                "recent_trials": [
                    self._compact_result(item)
                    for item in list(memory.get("recent_trials", []))[-3:]
                ],
                "accepted_trials": [
                    self._compact_result(item)
                    for item in list(memory.get("accepted_trials", []))[-2:]
                ],
            }
        return compact

    def _dataset_metadata(
        self, best_result: CandidateResult | None
    ) -> dict[str, object]:
        dataset_name = "uci_har"
        if best_result is not None:
            # Future-proofing for multi-dataset support if config later carries dataset name differently.
            dataset_name = "uci_har"
        if dataset_name == "uci_har":
            return UCIHARDataset().metadata()
        return {"dataset_name": dataset_name}

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
            role="orchestrator",
        )

    def _file_patches_schema(self) -> dict[str, object]:
        return {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "file_path": {"type": "string"},
                    "content": {"type": "string"},
                    "patch_mode": {"type": "string"},
                    "rationale": {"type": "string"},
                },
                "required": ["file_path", "content", "patch_mode", "rationale"],
            },
        }

    def _agent_prompt_prefix(
        self,
        *,
        agent_label: str,
        controls_text: str,
        guide_key: str,
        memory_key: str,
        context: dict[str, object],
        orchestrator_plan: dict[str, object],
    ) -> str:
        return (
            f"You are {agent_label} in an IMU HAR auto-research system.\n"
            f"{controls_text}\n\n"
            f"{agent_label} guide:\n{self.guides[guide_key]}\n\n"
            f"Orchestrator strategy:\n{json.dumps(orchestrator_plan, indent=2)}\n\n"
            f"{agent_label} local memory:\n"
            f"{json.dumps(context['agent_memories'][memory_key], indent=2)}\n\n"
            f"Round context:\n{json.dumps(context, indent=2)}\n\n"
        )

    def _model_agent_prompt(
        self,
        context: dict[str, object],
        orchestrator_plan: dict[str, object],
    ) -> str:
        rewrite_preference = self._rewrite_preference_note(int(context["round_id"]))
        return (
            self._agent_prompt_prefix(
                agent_label="ModelAgent",
                controls_text=(
                    "You only control model_name, learning_rate, and model-side params.\n"
                    "Propose one model-side mutation for this round.\n"
                    "Do not change augmentation or data settings."
                ),
                guide_key="model_agent",
                memory_key="model_agent",
                context=context,
                orchestrator_plan=orchestrator_plan,
            )
            + "Common available model families include {deepconvlstm, deepconvlstmattention, hart, tinierhar, tinyhar, spectra, cnn_gru, cnn, tiny_transformer, gru, custom}.\n"
            + "Treat those families as stable anchors, not as the whole point of the search. "
            + "Use them when they fit, but put most of your expertise into model-side helper logic, compatibility rules, and bounded rewrites inside the owned module.\n"
            + "You may also set model_params such as hidden_dim, depth, temporal_pool, spectral_bins, use_interactions, use_multiscale, and attention_mix.\n"
            + "Keep the rationale short and trend-aware.\n"
            + "Use `logic_expression` only for trivial direct overrides.\n"
            + "Prefer `helper_code` when the decision depends on baseline context, compatibility, guardrails, or conditional logic.\n"
            + "Do not behave like a dropdown picker. "
            + "If the important idea is a local model heuristic rather than a family switch, keep the family stable and express the novelty through `helper_code` or `function_rewrite`.\n"
            + rewrite_preference
            + "When the main mutation is family-dependent parameter compatibility, prefer defining "
            + "`model_params_override(...)` inside `function_rewrite` rather than only returning static model_params.\n"
            + "In particular, use `model_params_override(...)` when hidden_dim, depth, temporal_pool, spectral_bins, "
            + "use_interactions, use_multiscale, or attention_mix should depend on the current baseline context.\n"
            + "For deeper code evolution, you may also provide `function_rewrite` as a full Python function definition for "
            + "`bounded_learning_rate_override(base_experiment, target_name, baseline_name, baseline_lr, proposed_lr, config)`. "
            + "Use this when you want to rewrite model learning-rate bounding logic itself inside the owned module. "
            + "Return only a float learning rate or `None` to fall back.\n"
            + "The same `function_rewrite` block may also define "
            + "`model_params_override(base_experiment, target_name, proposed_params, config)` "
            + "when you want to rewrite model-parameter compatibility logic itself inside the owned module. "
            + "That function must return only a dict of model param overrides or `None` to fall back.\n"
            + "For a genuinely new bounded model-side representation, set `model_name` to `custom` and define "
            + "`custom_model_features(window, base_features, params)` in `function_rewrite`. "
            + "It may include small helper functions in the same block, must return a finite numeric feature list, "
            + "and should be deterministic and shape-stable across windows. "
            + "If the custom code errors or returns invalid features, the harness falls back to the standard feature vector.\n"
            + "Optionally provide a short Python expression in `logic_expression` that returns a dict of model-field overrides. "
            + "The expression may reference `base_experiment` and `config`. Return only owned-section updates such as "
            + '`{"model": {"name": ..., "learning_rate": ..., "params": {...}}}`. Use exact keys `name` and `learning_rate`. '
            + "You may also return a lambda `(lambda base_experiment, config: {...})`. Use an empty string when unnecessary.\n"
            + "Optionally provide `helper_code` containing small Python helper functions plus a top-level "
            + "`llm_choose_patch(base_experiment, config)` function. That function must return only owned model updates. "
            + "Keep helper_code short, self-contained, and limited to model-side reasoning. "
            + "If you emit helper_code, it should usually be the primary mechanism and `logic_expression` can be empty.\n"
            + "Prefer owned-module reasoning over menu hopping: use code when your idea is about architecture compatibility, family-dependent defaults, "
            + "or training-shape heuristics rather than just swapping one named family for another.\n"
        )

    def _augmentation_agent_prompt(
        self,
        context: dict[str, object],
        orchestrator_plan: dict[str, object],
    ) -> str:
        rewrite_preference = self._rewrite_preference_note(int(context["round_id"]))
        return (
            self._agent_prompt_prefix(
                agent_label="AugmentationAgent",
                controls_text=(
                    "You only control augmentation and strength.\n"
                    "Propose one augmentation-side mutation for this round.\n"
                    "Do not change model or data settings."
                ),
                guide_key="augmentation_agent",
                memory_key="augmentation_agent",
                context=context,
                orchestrator_plan=orchestrator_plan,
            )
            + "Common stable augmentation families include {none, rotation, time_warp, jitter, mixup, scaling, drift, permutation, magnitude_warp, custom}.\n"
            + "Common strengths include {weak, medium, strong}.\n"
            + "Treat those labels as anchors for the runtime, not as the main source of expertise. "
            + "Put the interesting reasoning into augmentation-side parameter logic, compatibility rules, and bounded rewrites inside the owned module.\n"
            + "Keep the rationale short and trend-aware.\n"
            + "Use `logic_expression` only for trivial direct overrides.\n"
            + "Prefer `helper_code` when the choice depends on baseline context, compatibility, guardrails, or conditional logic.\n"
            + "Do not behave like a menu selector. "
            + "If the important move is a context-sensitive perturbation policy, keep the family stable and express the novelty through `helper_code` or `function_rewrite`.\n"
            + rewrite_preference
            + "When the main mutation is augmentation-family or strength-dependent parameter compatibility, prefer defining "
            + "`augmentation_params_override(...)` inside `function_rewrite` rather than only relying on static defaults.\n"
            + "Use `augmentation_params_override(...)` when rotation angle, jitter std, time-warp compression, scaling span, "
            + "drift magnitude, permutation segments, magnitude-warp span, or mixup blend should depend on the current baseline context.\n"
            + "For deeper code evolution, you may also provide `function_rewrite` as a full Python function definition for "
            + "`compatibility_strength_override(base_experiment, target_name, requested_strength, config)`. "
            + "Use this when you want to rewrite augmentation strength compatibility logic itself inside the owned module. "
            + "Return only a valid strength string or `None` to fall back.\n"
            + "The same `function_rewrite` block may also define "
            + "`augmentation_params_override(base_experiment, target_name, target_strength, proposed_params, config)` "
            + "when you want to rewrite augmentation-parameter compatibility logic itself inside the owned module. "
            + "That function must return only a dict of augmentation param overrides or `None` to fall back.\n"
            + "For a genuinely new augmentation algorithm, set `augmentation` to `custom` and define "
            + "`custom_augment_window(window, strength, params, rng)` in `function_rewrite`. "
            + "It may include small helper functions in the same block, must return a window with the same shape, "
            + "and should use only bounded deterministic math/list operations plus `rng.uniform(...)` for randomness. "
            + "If the custom code errors or returns an invalid shape, the harness falls back to the unmodified window.\n"
            + "Optionally provide a short Python expression in `logic_expression` that returns a dict of augmentation-field overrides. "
            + "The expression may reference `base_experiment` and `config`. Return only owned-section updates such as "
            + '`{"augmentation": {"name": ..., "strength": ...}}`. Use exact keys `name` and `strength`. '
            + "You may also return a lambda `(lambda base_experiment, config: {...})`. Use an empty string when unnecessary.\n"
            + "Optionally provide `helper_code` containing small Python helper functions plus a top-level "
            + "`llm_choose_patch(base_experiment, config)` function. That function must return only owned augmentation updates. "
            + "Keep helper_code short, self-contained, and limited to augmentation-side reasoning. "
            + "If you emit helper_code, it should usually be the primary mechanism and `logic_expression` can be empty.\n"
            + "Prefer owned-module reasoning over family roulette: use code when your idea is about realism, robustness, parameter shaping, or compatibility "
            + "with the current model/data context.\n"
        )

    def _data_agent_prompt(
        self,
        context: dict[str, object],
        orchestrator_plan: dict[str, object],
    ) -> str:
        rewrite_preference = self._rewrite_preference_note(int(context["round_id"]))
        return (
            self._agent_prompt_prefix(
                agent_label="DataAgent",
                controls_text=(
                    "You only control preprocessing and window_size.\n"
                    "Propose one data-side mutation for this round.\n"
                    "Do not change model or augmentation settings."
                ),
                guide_key="data_agent",
                memory_key="data_agent",
                context=context,
                orchestrator_plan=orchestrator_plan,
            )
            + "Common stable preprocessing regimes include {raw, normalized, normalized+gravity_filter, normalized+magnitude_channels, custom}.\n"
            + "Common stable window sizes include {64, 96, 128, 192, 256}.\n"
            + "Treat those settings as anchors for the runtime, not as the whole search space. "
            + "Put the interesting reasoning into data-side helper logic, compatibility rules, and bounded rewrites inside the owned module.\n"
            + "Keep the rationale short and trend-aware.\n"
            + "Use `logic_expression` only for trivial direct overrides.\n"
            + "Prefer `helper_code` when the choice depends on baseline context, compatibility, guardrails, or conditional logic.\n"
            + "Do not behave like a fixed menu chooser. "
            + "If the real idea is a representation heuristic, keep the top-level setting stable and express the novelty through `helper_code` or `function_rewrite`.\n"
            + rewrite_preference
            + "When the main mutation is preprocessing/window-dependent data parameter compatibility, prefer defining "
            + "`data_params_override(...)` inside `function_rewrite` rather than only relying on static defaults.\n"
            + "Use `data_params_override(...)` when channel layout, resampling mode, gravity scale, or magnitude toggles "
            + "should depend on the current baseline context.\n"
            + "For deeper code evolution, you may also provide `function_rewrite` as a full Python function definition for "
            + "`choose_window_override(base_experiment, requested_window, config)`. "
            + "Use this when you want to rewrite data window-selection logic itself inside the owned module. "
            + "Return only an integer window size or `None` to fall back.\n"
            + "The same `function_rewrite` block may also define "
            + "`data_params_override(base_experiment, target_preprocessing, target_window, proposed_params, config)` "
            + "when you want to rewrite data-parameter compatibility logic itself inside the owned module. "
            + "That function must return only a dict of data param overrides or `None` to fall back.\n"
            + "For a genuinely new bounded preprocessing transform, set `preprocessing` to `custom` and define "
            + "`custom_prepare_window(window, params)` in `function_rewrite`. "
            + "It may include small helper functions in the same block, must return a finite numeric window with the same shape, "
            + "and should preserve label semantics. "
            + "If the custom code errors or returns an invalid shape, the harness falls back to the standard preprocessed window.\n"
            + "Optionally provide a short Python expression in `logic_expression` that returns a dict of data-field overrides. "
            + "The expression may reference `base_experiment` and `config`. Return only owned-section updates such as "
            + '`{"data": {"preprocessing": ..., "window_size": ...}}`. Use exact keys `preprocessing` and `window_size`. '
            + "You may also return a lambda `(lambda base_experiment, config: {...})`. Use an empty string when unnecessary.\n"
            + "Optionally provide `helper_code` containing small Python helper functions plus a top-level "
            + "`llm_choose_patch(base_experiment, config)` function. That function must return only owned data updates. "
            + "Keep helper_code short, self-contained, and limited to data-side reasoning. "
            + "If you emit helper_code, it should usually be the primary mechanism and `logic_expression` can be empty.\n"
            + "Prefer owned-module reasoning over cosmetic setting churn: use code when your idea is about representation quality, window compatibility, "
            + "or preprocessing guardrails rather than just flipping to a neighboring preset.\n"
        )

    def _plan_model_agent(
        self,
        context: dict[str, object],
        orchestrator_plan: dict[str, object],
    ) -> ModelPlanHint:
        prompt = self._model_agent_prompt(context, orchestrator_plan)
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
                "file_patches": self._file_patches_schema(),
            },
            "required": [
                "model_name",
                "learning_rate",
                "model_params",
                "rationale",
                "logic_expression",
                "helper_code",
                "function_rewrite",
                "file_patches",
            ],
        }
        payload = self._call_json_schema(
            name="autohar_model_agent_plan",
            prompt=prompt,
            schema=schema,
            role="model_agent",
        )
        original_payload = dict(payload)
        payload = self._apply_model_exploration_pressure(context, payload)
        payload = self._prefer_model_param_rewrite(
            round_id=int(context["round_id"]),
            payload=payload,
        )
        payload = self._finalize_model_payload(
            payload=payload, original_payload=original_payload
        )
        payload["file_patches"] = [
            FilePatch(**fp) for fp in payload.get("file_patches", [])
        ]
        return ModelPlanHint(**payload)

    def _plan_augmentation_agent(
        self,
        context: dict[str, object],
        orchestrator_plan: dict[str, object],
    ) -> AugmentationPlanHint:
        prompt = self._augmentation_agent_prompt(context, orchestrator_plan)
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
                "file_patches": self._file_patches_schema(),
            },
            "required": [
                "augmentation",
                "strength",
                "rationale",
                "logic_expression",
                "helper_code",
                "function_rewrite",
                "file_patches",
            ],
        }
        payload = self._call_json_schema(
            name="autohar_augmentation_agent_plan",
            prompt=prompt,
            schema=schema,
            role="augmentation_agent",
        )
        original_payload = dict(payload)
        payload = self._apply_augmentation_exploration_pressure(context, payload)
        payload = self._prefer_augmentation_param_rewrite(
            round_id=int(context["round_id"]),
            payload=payload,
        )
        payload = self._reconcile_augmentation_logic(
            payload=payload, original_payload=original_payload
        )
        payload = self._finalize_augmentation_payload(
            payload=payload, original_payload=original_payload
        )
        payload["file_patches"] = [
            FilePatch(**fp) for fp in payload.get("file_patches", [])
        ]
        return AugmentationPlanHint(**payload)

    def _plan_data_agent(
        self,
        context: dict[str, object],
        orchestrator_plan: dict[str, object],
    ) -> DataPlanHint:
        prompt = self._data_agent_prompt(context, orchestrator_plan)
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
                "file_patches": self._file_patches_schema(),
            },
            "required": [
                "preprocessing",
                "window_size",
                "rationale",
                "logic_expression",
                "helper_code",
                "function_rewrite",
                "file_patches",
            ],
        }
        payload = self._call_json_schema(
            name="autohar_data_agent_plan",
            prompt=prompt,
            schema=schema,
            role="data_agent",
        )
        original_payload = dict(payload)
        payload = self._apply_data_exploration_pressure(context, payload)
        payload = self._prefer_data_param_rewrite(
            round_id=int(context["round_id"]),
            payload=payload,
        )
        payload = self._reconcile_data_logic(
            payload=payload, original_payload=original_payload
        )
        payload = self._finalize_data_payload(
            payload=payload, original_payload=original_payload
        )
        payload["file_patches"] = [
            FilePatch(**fp) for fp in payload.get("file_patches", [])
        ]
        return DataPlanHint(**payload)

    def _call_json_schema(
        self,
        name: str,
        prompt: str,
        schema: dict[str, object],
        *,
        role: str,
    ) -> dict[str, object]:
        client, model_name, api_mode = self._client_for_role(role)
        if api_mode == "opencode":
            current_prompt = prompt
            for attempt in range(3):
                try:
                    content = self._opencode_json_completion(
                        role=role,
                        model_name=model_name,
                        prompt=current_prompt,
                        schema=schema,
                    )
                    return self._parse_json_response(content)
                except Exception:
                    if attempt < 2:
                        current_prompt = (
                            "Return a minimal valid JSON payload only. "
                            "Do not include explanations.\n\n"
                            f"{prompt}"
                        )
                        continue
                    return self._fallback_json_payload(name=name)
        if api_mode == "chat":
            max_tokens = int(os.getenv("AUTOHAR_CHAT_MAX_TOKENS", "2048"))
            response_format_mode = (
                os.getenv("AUTOHAR_CHAT_RESPONSE_FORMAT", "text").strip().lower()
            )
            messages = [
                {
                    "role": "system",
                    "content": (
                        "Return only valid JSON. Do not use markdown fences. "
                        "Follow the requested schema exactly."
                    ),
                },
                {"role": "user", "content": prompt},
            ]
            last_error: Exception | None = None
            for attempt in range(3):
                try:
                    response_format: dict[str, object]
                    if response_format_mode == "json_schema":
                        response_format = {
                            "type": "json_schema",
                            "json_schema": {
                                "name": name,
                                "schema": schema,
                            },
                        }
                    else:
                        response_format = {"type": "text"}
                    response = client.chat.completions.create(
                        model=model_name,
                        messages=messages,
                        response_format=response_format,
                        temperature=0.0,
                        max_tokens=max_tokens,
                    )
                except Exception as exc:  # noqa: BLE001
                    last_error = exc
                    if attempt < 2:
                        messages = [
                            messages[0],
                            {
                                "role": "user",
                                "content": (
                                    "Return a minimal valid JSON payload for "
                                    f"{name}. Avoid explanations and omit optional detail."
                                ),
                            },
                        ]
                        continue
                    return self._fallback_json_payload(name=name)
                content = response.choices[0].message.content or "{}"
                try:
                    return self._parse_json_response(content)
                except json.JSONDecodeError as exc:
                    last_error = exc
                    messages.append(
                        {
                            "role": "assistant",
                            "content": content,
                        }
                    )
                    messages.append(
                        {
                            "role": "user",
                            "content": (
                                "That response was invalid JSON. Return the same answer again as strict valid JSON only."
                            ),
                        }
                    )
            return self._fallback_json_payload(name=name)
        response = client.responses.create(
            model=model_name,
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

    def _opencode_json_completion(
        self,
        *,
        role: str,
        model_name: str,
        prompt: str,
        schema: dict[str, object],
    ) -> str:
        role_key = self.ROLE_KEYS.get(role, role.upper())
        binary = os.getenv(
            f"AUTOHAR_OPENCODE_{role_key}_BIN",
            os.getenv(
                "AUTOHAR_OPENCODE_BIN", os.path.expanduser("~/.opencode/bin/opencode")
            ),
        )
        full_prompt = (
            "Return only valid JSON. Do not use markdown fences.\n"
            "Follow this JSON schema exactly:\n"
            f"{json.dumps(schema, indent=2)}\n\n"
            f"{prompt}"
        )
        command = [
            binary,
            "run",
            "--format",
            "json",
            "--dir",
            str(Path.cwd()),
            "--model",
            model_name,
        ]
        variant = os.getenv(
            f"AUTOHAR_OPENCODE_{role_key}_VARIANT",
            os.getenv("AUTOHAR_OPENCODE_VARIANT", ""),
        ).strip()
        if variant:
            command.extend(["--variant", variant])
        if os.getenv("AUTOHAR_OPENCODE_PURE", "true").strip().lower() == "true":
            command.append("--pure")
        command.append(full_prompt)
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False,
        )
        output = "\n".join(
            chunk for chunk in (result.stdout.strip(), result.stderr.strip()) if chunk
        )
        if result.returncode != 0:
            raise RuntimeError(output or "opencode command failed")
        text_events: list[str] = []
        for line in output.splitlines():
            line = line.strip()
            if not line or not line.startswith("{"):
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if event.get("type") == "text":
                part = event.get("part", {})
                if isinstance(part, dict) and isinstance(part.get("text"), str):
                    text_events.append(part["text"])
        if not text_events:
            raise RuntimeError("No text response found in opencode output")
        return "\n".join(text_events)

    def _default_opencode_model(self) -> str:
        configured = os.getenv("AUTOHAR_OPENCODE_MODEL", "").strip()
        if configured:
            return configured
        preferred_local = [
            "lmstudio/qwen/qwen3-30b-a3b-2507",
            "lmstudio/qwen/qwen3-coder-30b",
            "lmstudio/openai/gpt-oss-20b",
        ]
        models = self._opencode_models()
        local_models = [model for model in models if model.startswith("lmstudio/")]
        for preferred in preferred_local:
            if preferred in local_models:
                return preferred
        if local_models:
            return local_models[0]
        for preferred in preferred_local:
            return preferred
        preferred_remote = [
            "opencode/gpt-5-nano",
            "opencode/big-pickle",
            "opencode/nemotron-3-super-free",
        ]
        for preferred in preferred_remote:
            if preferred in models:
                return preferred
        return self.model or "opencode/gpt-5-nano"

    def _opencode_models(self) -> list[str]:
        if self._opencode_model_cache is not None:
            return self._opencode_model_cache
        binary = os.getenv(
            "AUTOHAR_OPENCODE_BIN", os.path.expanduser("~/.opencode/bin/opencode")
        )
        try:
            result = subprocess.run(
                [binary, "models"],
                capture_output=True,
                text=True,
                check=False,
            )
            if result.returncode != 0:
                self._opencode_model_cache = []
                return self._opencode_model_cache
            self._opencode_model_cache = [
                line.strip()
                for line in result.stdout.splitlines()
                if line.strip()
            ]
        except Exception:
            self._opencode_model_cache = []
        return self._opencode_model_cache

    def _parse_json_response(self, content: str) -> dict[str, object]:
        content = content.strip()
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            start = content.find("{")
            end = content.rfind("}")
            if start != -1 and end != -1 and end > start:
                return json.loads(content[start : end + 1])
            raise

    def _fallback_json_payload(self, name: str) -> dict[str, object]:
        if name == "autohar_orchestrator_plan":
            return {
                "summary": "Fallback local planner strategy: run conservative specialist mutations and let validation decide.",
                "priority_order": ["model", "data", "augmentation"],
                "exploration_mode": "fallback_explore",
                "combination_bias": "pairwise",
                "risk_note": "Used because the local LLM returned malformed JSON after retries.",
            }
        if name == "autohar_model_agent_plan":
            return {
                "model_name": "deepconvlstm",
                "learning_rate": 0.0003,
                "model_params": {
                    "hidden_dim": 80,
                    "depth": 2,
                    "temporal_pool": "avg",
                    "spectral_bins": 6,
                    "use_interactions": True,
                    "use_multiscale": True,
                    "attention_mix": 0.55,
                },
                "rationale": "Fallback model proposal after malformed local LLM JSON.",
                "logic_expression": "",
                "helper_code": "",
                "function_rewrite": "",
            }
        if name == "autohar_augmentation_agent_plan":
            return {
                "augmentation": "rotation",
                "strength": "weak",
                "rationale": "Fallback augmentation proposal after malformed local LLM JSON.",
                "logic_expression": "",
                "helper_code": "",
                "function_rewrite": "",
            }
        if name == "autohar_data_agent_plan":
            return {
                "preprocessing": "raw",
                "window_size": 128,
                "rationale": "Fallback data proposal after malformed local LLM JSON.",
                "logic_expression": "",
                "helper_code": "",
                "function_rewrite": "",
            }
        if name == "autohar_proposal_judgments":
            return {
                "model_agent": {
                    "mutation_size": "medium",
                    "novelty": "medium",
                    "plausibility": "medium",
                    "compatibility": "medium",
                    "evaluate_now": True,
                    "rationale": "Fallback judge approval after malformed local LLM JSON.",
                },
                "augmentation_agent": {
                    "mutation_size": "medium",
                    "novelty": "medium",
                    "plausibility": "medium",
                    "compatibility": "medium",
                    "evaluate_now": True,
                    "rationale": "Fallback judge approval after malformed local LLM JSON.",
                },
                "data_agent": {
                    "mutation_size": "medium",
                    "novelty": "medium",
                    "plausibility": "medium",
                    "compatibility": "medium",
                    "evaluate_now": True,
                    "rationale": "Fallback judge approval after malformed local LLM JSON.",
                },
            }
        return {}

    def _role_runtime_config(self, role: str) -> dict[str, str]:
        role_key = self.ROLE_KEYS.get(role, role.upper())
        backend = (
            os.getenv(
                f"AUTOHAR_{role_key}_BACKEND",
                os.getenv("AUTOHAR_LLM_BACKEND", "openai"),
            )
            .strip()
            .lower()
        )
        if backend == "opencode":
            return {
                "backend": backend,
                "model": os.getenv(
                    f"AUTOHAR_OPENCODE_{role_key}_MODEL",
                    os.getenv(
                        f"AUTOHAR_{role_key}_MODEL",
                        os.getenv(
                            "AUTOHAR_OPENCODE_MODEL",
                            self._default_opencode_model(),
                        ),
                    ),
                ),
                "base_url": os.getenv(
                    f"AUTOHAR_OPENCODE_{role_key}_BIN",
                    os.getenv(
                        "AUTOHAR_OPENCODE_BIN",
                        os.path.expanduser("~/.opencode/bin/opencode"),
                    ),
                ),
                "api_key": "",
                "api_mode": "run",
            }
        if backend == "lmstudio":
            return {
                "backend": backend,
                "model": os.getenv(
                    f"AUTOHAR_LMSTUDIO_{role_key}_MODEL",
                    os.getenv(
                        f"AUTOHAR_{role_key}_MODEL",
                        os.getenv("AUTOHAR_LMSTUDIO_MODEL", self.model),
                    ),
                ),
                "base_url": os.getenv(
                    f"AUTOHAR_LMSTUDIO_{role_key}_BASE_URL",
                    os.getenv(
                        f"AUTOHAR_{role_key}_BASE_URL",
                        os.getenv(
                            "AUTOHAR_LLM_BASE_URL",
                            self.base_url or "http://127.0.0.1:1234/v1",
                        ),
                    ),
                ),
                "api_key": os.getenv(
                    f"AUTOHAR_LMSTUDIO_{role_key}_API_KEY",
                    os.getenv(
                        f"AUTOHAR_{role_key}_API_KEY",
                        os.getenv(
                            "AUTOHAR_LLM_API_KEY",
                            self.default_api_key or "lm-studio",
                        ),
                    ),
                ),
                "api_mode": (
                    os.getenv(
                        f"AUTOHAR_LMSTUDIO_{role_key}_API_MODE",
                        os.getenv(f"AUTOHAR_{role_key}_API_MODE", "chat"),
                    )
                    .strip()
                    .lower()
                ),
            }
        return {
            "backend": backend,
            "model": os.getenv(
                f"AUTOHAR_OPENAI_{role_key}_MODEL",
                os.getenv(
                    f"AUTOHAR_{role_key}_MODEL",
                    os.getenv("AUTOHAR_OPENAI_MODEL", self.model),
                ),
            ),
            "base_url": os.getenv(
                f"AUTOHAR_OPENAI_{role_key}_BASE_URL",
                os.getenv(f"AUTOHAR_{role_key}_BASE_URL", self.base_url or ""),
            ),
            "api_key": os.getenv(
                f"AUTOHAR_OPENAI_{role_key}_API_KEY",
                os.getenv(
                    f"AUTOHAR_{role_key}_API_KEY",
                    self.default_api_key or os.getenv("OPENAI_API_KEY", ""),
                ),
            ),
            "api_mode": (
                os.getenv(
                    f"AUTOHAR_OPENAI_{role_key}_API_MODE",
                    os.getenv(f"AUTOHAR_{role_key}_API_MODE", self.api_mode),
                )
                .strip()
                .lower()
            ),
        }

    def _client_for_role(self, role: str) -> tuple[object | None, str, str]:
        runtime = self._role_runtime_config(role)
        backend = runtime["backend"]
        model_name = runtime["model"]
        base_url = runtime["base_url"]
        api_key = runtime["api_key"]
        api_mode = runtime["api_mode"]
        if OpenAI is None:
            raise RuntimeError("openai package is not installed")
        cache_key = (backend, base_url, api_key, api_mode)
        client = self._client_cache.get(cache_key)
        if client is None:
            client_kwargs: dict[str, object] = {}
            if base_url:
                client_kwargs["base_url"] = base_url
            if api_key:
                client_kwargs["api_key"] = api_key
            client = OpenAI(**client_kwargs)
            self._client_cache[cache_key] = client
        return client, model_name, api_mode

    def role_runtime_summary(self) -> dict[str, dict[str, str]]:
        summary: dict[str, dict[str, str]] = {}
        for role in (
            "orchestrator",
            "model_agent",
            "augmentation_agent",
            "data_agent",
            "judge",
        ):
            runtime = self._role_runtime_config(role)
            summary[role] = {
                "backend": runtime["backend"],
                "model": runtime["model"],
                "api_mode": runtime["api_mode"],
                "base_url": runtime["base_url"],
            }
        return summary

    def _proposal_payload(self, proposal: object) -> dict[str, object]:
        if isinstance(proposal, ModelPlanHint):
            return {
                "model_name": proposal.model_name,
                "learning_rate": proposal.learning_rate,
                "model_params": dict(proposal.model_params),
                "rationale": proposal.rationale,
                "logic_expression": proposal.logic_expression,
                "helper_code": proposal.helper_code,
                "function_rewrite": proposal.function_rewrite,
            }
        if isinstance(proposal, AugmentationPlanHint):
            return {
                "augmentation": proposal.augmentation,
                "strength": proposal.strength,
                "rationale": proposal.rationale,
                "logic_expression": proposal.logic_expression,
                "helper_code": proposal.helper_code,
                "function_rewrite": proposal.function_rewrite,
            }
        if isinstance(proposal, DataPlanHint):
            return {
                "preprocessing": proposal.preprocessing,
                "window_size": proposal.window_size,
                "rationale": proposal.rationale,
                "logic_expression": proposal.logic_expression,
                "helper_code": proposal.helper_code,
                "function_rewrite": proposal.function_rewrite,
            }
        from dataclasses import asdict

        return (
            asdict(proposal)
            if hasattr(proposal, "__dataclass_fields__")
            else {"raw": str(proposal)}
        )

    def _judgment_schema(self) -> dict[str, object]:
        return {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "mutation_size": {"type": "string"},
                "novelty": {"type": "string"},
                "plausibility": {"type": "string"},
                "compatibility": {"type": "string"},
                "evaluate_now": {"type": "boolean"},
                "rationale": {"type": "string"},
            },
            "required": [
                "mutation_size",
                "novelty",
                "plausibility",
                "compatibility",
                "evaluate_now",
                "rationale",
            ],
        }

    def _load_guides(self) -> dict[str, str]:
        guide_root = Path.cwd() / "agent_guides"
        return {
            "orchestrator": (guide_root / "orchestrator.md").read_text(
                encoding="utf-8"
            ),
            "model_agent": (guide_root / "model_agent.md").read_text(encoding="utf-8"),
            "augmentation_agent": (guide_root / "augmentation_agent.md").read_text(
                encoding="utf-8"
            ),
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
                scores = [
                    result.val_score
                    for result in recent
                    if result.agent_name == agent_name
                ]
                best_by_agent[agent_name] = round(max(scores), 4)
                avg_by_agent[agent_name] = round(sum(scores) / len(scores), 4)
        plateau = False
        if len(accepted_history) >= 3:
            recent_accepts = [result.val_score for result in accepted_history[-3:]]
            plateau = max(recent_accepts) - min(recent_accepts) < 0.01
        latest_round = max((result.round_id for result in history), default=0)
        last_accept_round = max(
            (result.round_id for result in accepted_history), default=0
        )
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
                label = str(
                    result.artifact_paths.get(
                        "selected_combination", "model+augmentation+data"
                    )
                )
            else:
                label = result.agent_name
            if label in by_label:
                by_label[label].append(result.val_score)
        for result in accepted_history:
            if result.agent_name == "orchestrator":
                label = str(
                    result.artifact_paths.get(
                        "selected_combination", "model+augmentation+data"
                    )
                )
            else:
                label = result.agent_name
            if label in accepted_by_label:
                accepted_by_label[label].append(result.val_score)
        included_map = {
            "model+augmentation": ["model_agent", "augmentation_agent"],
            "model+data": ["model_agent", "data_agent"],
            "augmentation+data": ["augmentation_agent", "data_agent"],
            "model+augmentation+data": [
                "model_agent",
                "augmentation_agent",
                "data_agent",
            ],
        }
        for round_id, round_results in round_groups.items():
            singles = {
                result.agent_name: result.val_score
                for result in round_results
                if result.agent_name
                in {"model_agent", "augmentation_agent", "data_agent"}
            }
            for result in round_results:
                if result.agent_name != "orchestrator":
                    continue
                label = str(
                    result.artifact_paths.get(
                        "selected_combination", "model+augmentation+data"
                    )
                )
                included = included_map.get(label)
                if not included:
                    continue
                component_best = max(
                    (singles.get(name, float("-inf")) for name in included),
                    default=float("-inf"),
                )
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
            label: round(max(scores), 4) for label, scores in by_label.items() if scores
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
        synergy_sorted = sorted(
            synergy_records, key=lambda item: item["synergy"], reverse=True
        )
        compatible_patterns = [
            item for item in synergy_sorted if float(item["synergy"]) > 0
        ][:3]
        conflicting_patterns = [
            item
            for item in sorted(synergy_records, key=lambda item: item["synergy"])
            if float(item["synergy"]) < 0
        ][:3]
        return {
            "recent_best_by_combination": recent_best,
            "recent_avg_by_combination": recent_avg,
            "accepted_best_by_combination": accepted_best,
            "compatible_patterns": compatible_patterns,
            "conflicting_patterns": conflicting_patterns,
        }

    def _rewrite_preference_note(self, round_id: int) -> str:
        if round_id <= 1:
            return "In round 1, bounded function rewrites are optional; prefer them only when they add real value.\n"
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
        scoped_history = [
            result for result in history if result.agent_name == agent_name
        ]
        accepted_scoped = [
            result for result in accepted_history if result.agent_name == agent_name
        ]
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
        latest_round = max((result.round_id for result in history), default=0)
        last_accept_round = max(
            (result.round_id for result in accepted_history), default=0
        )
        stagnation_rounds = max(0, latest_round - last_accept_round)

        model_values = [
            result.config.model_name
            for result in history
            if result.agent_name == "model_agent"
        ]
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
            if stagnation_rounds >= 10:
                params = dict(payload.get("model_params") or {})
                hidden_dim = int(params.get("hidden_dim", 96))
                depth = int(params.get("depth", 2))
                params["hidden_dim"] = max(hidden_dim, int(round(hidden_dim * 1.2)))
                params["depth"] = min(6, depth + 1)
                params["attention_mix"] = max(
                    0.35, min(0.9, float(params.get("attention_mix", 0.5)) + 0.08)
                )
                payload["model_params"] = params
                payload["rationale"] = (
                    f"{payload.get('rationale', '').strip()} "
                    "[forced larger model mutation after prolonged stagnation]"
                ).strip()
            return payload
        if stagnation_rounds < 5 and not plateau and repeat_streak < 3:
            return payload
        recent_models = [
            str(item["config"]["model_name"])
            for item in context["agent_memories"]["model_agent"].get(
                "recent_trials", []
            )
            if isinstance(item, dict)
        ]
        candidate_order = self.MODEL_FAMILIES
        if stagnation_rounds >= 5:
            candidate_order = [
                candidate
                for candidate in self.MODEL_FAMILIES
                if candidate
                not in {baseline_model, "deepconvlstm", "deepconvlstmattention"}
            ] + self.MODEL_FAMILIES
        for candidate in candidate_order:
            if candidate != baseline_model and candidate not in recent_models[-3:]:
                payload["model_name"] = candidate
                if stagnation_rounds >= 10:
                    current_params = dict(payload.get("model_params") or {})
                    current_params["hidden_dim"] = max(
                        int(current_params.get("hidden_dim", 96)), 128
                    )
                    current_params["depth"] = min(
                        6, max(int(current_params.get("depth", 2)), 3)
                    )
                    current_params["attention_mix"] = max(
                        0.35,
                        min(
                            0.9, float(current_params.get("attention_mix", 0.5)) + 0.08
                        ),
                    )
                    payload["model_params"] = current_params
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
        current = (
            str(payload.get("augmentation", "")),
            str(payload.get("strength", "")),
        )
        if current != (baseline_aug, baseline_strength):
            if stagnation_rounds >= 10:
                payload["strength"] = (
                    "medium"
                    if str(payload.get("strength", "")) == "weak"
                    else str(payload.get("strength", "medium"))
                )
                payload["rationale"] = (
                    f"{payload.get('rationale', '').strip()} "
                    "[forced real augmentation-family/strength shift after prolonged stagnation]"
                ).strip()
            return payload
        if stagnation_rounds < 5 and not plateau and repeat_streak < 3:
            return payload
        recent_values = [
            (
                str(item["config"]["augmentation"]),
                str(item["config"]["augmentation_strength"]),
            )
            for item in context["agent_memories"]["augmentation_agent"].get(
                "recent_trials", []
            )
            if isinstance(item, dict)
        ]
        augmentation_order = self.AUGMENTATIONS
        if stagnation_rounds >= 5:
            augmentation_order = [
                aug for aug in self.AUGMENTATIONS if aug not in {baseline_aug, "none"}
            ] + self.AUGMENTATIONS
        for augmentation in augmentation_order:
            for strength in self.AUGMENTATION_STRENGTHS:
                candidate = (augmentation, strength)
                if (
                    candidate != (baseline_aug, baseline_strength)
                    and candidate not in recent_values[-3:]
                ):
                    payload["augmentation"] = augmentation
                    payload["strength"] = (
                        "medium"
                        if stagnation_rounds >= 10 and strength == "weak"
                        else strength
                    )
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
        current = (
            str(payload.get("preprocessing", "")),
            int(payload.get("window_size", 128)),
        )
        if current != (baseline_pre, baseline_window):
            if stagnation_rounds >= 10:
                payload["window_size"] = (
                    192
                    if int(payload.get("window_size", 128)) == baseline_window
                    else int(payload.get("window_size", 128))
                )
                payload["rationale"] = (
                    f"{payload.get('rationale', '').strip()} "
                    "[forced larger data-regime shift after prolonged stagnation]"
                ).strip()
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
            preprocessing_order = [
                mode for mode in self.PREPROCESSING_MODES if mode != baseline_pre
            ] + self.PREPROCESSING_MODES
            window_order = [
                size for size in self.WINDOW_SIZES if size != baseline_window
            ] + self.WINDOW_SIZES
        for preprocessing in preprocessing_order:
            for window_size in window_order:
                candidate = (preprocessing, window_size)
                if (
                    candidate != (baseline_pre, baseline_window)
                    and candidate not in recent_values[-3:]
                ):
                    payload["preprocessing"] = preprocessing
                    payload["window_size"] = (
                        192
                        if stagnation_rounds >= 10 and window_size == baseline_window
                        else window_size
                    )
                    payload["rationale"] = (
                        f"{payload.get('rationale', '').strip()} "
                        f"[exploration pressure: diversify away from repeated/stagnant baseline data setting {baseline_pre}:{baseline_window} -> {preprocessing}:{window_size}]"
                    ).strip()
                    return payload
        return payload

    def _apply_stagnation_overrides_to_orchestrator_plan(
        self,
        context: dict[str, object],
        orchestrator_plan: dict[str, object],
    ) -> dict[str, object]:
        pressure = dict(context.get("exploration_pressure", {}))
        stagnation_rounds = int(pressure.get("stagnation_rounds", 0))
        if stagnation_rounds < 10:
            return orchestrator_plan
        updated = dict(orchestrator_plan)
        updated["exploration_mode"] = "forced_diversify"
        updated["priority_order"] = ["model", "augmentation", "data"]
        updated["combination_bias"] = "pairwise"
        updated["risk_note"] = (
            f"{str(updated.get('risk_note', '')).strip()} "
            "[stagnation override: after 10+ non-improving rounds, require one larger model mutation, one real augmentation-family shift, and one real data-regime shift.]"
        ).strip()
        updated["summary"] = (
            f"{str(updated.get('summary', '')).strip()} "
            "[stagnation override active: broaden specialist mutations instead of only protecting the current winner.]"
        ).strip()
        return updated

    def _relax_judgments(
        self,
        context: dict[str, object],
        payload: dict[str, object],
    ) -> dict[str, object]:
        pressure = dict(context.get("exploration_pressure", {}))
        stagnation_rounds = int(pressure.get("stagnation_rounds", 0))
        for judgment in payload.values():
            if not isinstance(judgment, dict):
                continue
            plausibility = str(judgment.get("plausibility", "medium"))
            compatibility = str(judgment.get("compatibility", "medium"))
            novelty = str(judgment.get("novelty", "medium"))
            if plausibility != "low" and compatibility != "low":
                judgment["evaluate_now"] = True
            if (
                stagnation_rounds >= 10
                and novelty == "low"
                and judgment.get("mutation_size") == "small"
            ):
                judgment["mutation_size"] = "medium"
                judgment["rationale"] = (
                    f"{str(judgment.get('rationale', '')).strip()} "
                    "[judge relaxation: keep plausible proposals in play during prolonged stagnation.]"
                ).strip()
        return payload

    def _finalize_model_payload(
        self,
        *,
        payload: dict[str, object],
        original_payload: dict[str, object],
    ) -> dict[str, object]:
        params = dict(payload.get("model_params") or {})
        original_model = str(original_payload.get("model_name", ""))
        final_model = str(payload.get("model_name", ""))
        original_lr = float(
            original_payload.get("learning_rate", payload.get("learning_rate", 0.0))
        )
        final_lr = float(payload.get("learning_rate", original_lr))
        tags = self._extract_rationale_tags(str(payload.get("rationale", "")))
        change_note = ""
        if final_model != original_model or abs(final_lr - original_lr) > 1e-12:
            change_note = (
                f" postprocessed from {original_model or 'unknown'}@{original_lr:.5g}."
            )
        payload["rationale"] = (
            f"Final model proposal: target {final_model} at lr={final_lr:.5g}; "
            f"hidden_dim={int(params.get('hidden_dim', 0))}, depth={int(params.get('depth', 0))}, "
            f"temporal_pool={params.get('temporal_pool', 'unknown')}, spectral_bins={int(params.get('spectral_bins', 0))}."
            f"{change_note}"
            f"{self._logic_note(payload)}"
            f"{self._append_tags(tags)}"
        ).strip()
        return payload

    def _finalize_augmentation_payload(
        self,
        *,
        payload: dict[str, object],
        original_payload: dict[str, object],
    ) -> dict[str, object]:
        original_aug = str(original_payload.get("augmentation", ""))
        original_strength = str(original_payload.get("strength", ""))
        final_aug = str(payload.get("augmentation", ""))
        final_strength = str(payload.get("strength", ""))
        tags = self._extract_rationale_tags(str(payload.get("rationale", "")))
        change_note = ""
        if (final_aug, final_strength) != (original_aug, original_strength):
            change_note = f" postprocessed from {original_aug or 'unknown'}:{original_strength or 'unknown'}."
        payload["rationale"] = (
            f"Final augmentation proposal: target {final_aug}:{final_strength}."
            f"{change_note}"
            f"{self._logic_note(payload)}"
            f"{self._append_tags(tags)}"
        ).strip()
        return payload

    def _finalize_data_payload(
        self,
        *,
        payload: dict[str, object],
        original_payload: dict[str, object],
    ) -> dict[str, object]:
        original_pre = str(original_payload.get("preprocessing", ""))
        original_window = int(
            original_payload.get("window_size", payload.get("window_size", 128))
        )
        final_pre = str(payload.get("preprocessing", ""))
        final_window = int(payload.get("window_size", original_window))
        tags = self._extract_rationale_tags(str(payload.get("rationale", "")))
        change_note = ""
        if (final_pre, final_window) != (original_pre, original_window):
            change_note = (
                f" postprocessed from {original_pre or 'unknown'}:{original_window}."
            )
        payload["rationale"] = (
            f"Final data proposal: target preprocessing={final_pre}, window={final_window}."
            f"{change_note}"
            f"{self._logic_note(payload)}"
            f"{self._append_tags(tags)}"
        ).strip()
        return payload

    def _reconcile_augmentation_logic(
        self,
        *,
        payload: dict[str, object],
        original_payload: dict[str, object],
    ) -> dict[str, object]:
        final_aug = str(payload.get("augmentation", "")).strip().lower()
        original_aug = str(original_payload.get("augmentation", "")).strip().lower()
        if final_aug == original_aug:
            return payload
        if final_aug == "none":
            payload["logic_expression"] = ""
            payload["helper_code"] = ""
            payload["function_rewrite"] = ""
            return payload
        function_rewrite = str(payload.get("function_rewrite", ""))
        helper_code = str(payload.get("helper_code", ""))
        logic_expression = str(payload.get("logic_expression", ""))
        combined = " ".join(
            [function_rewrite.lower(), helper_code.lower(), logic_expression.lower()]
        )
        if original_aug and original_aug in combined and final_aug not in combined:
            payload["logic_expression"] = ""
            payload["helper_code"] = ""
            payload["function_rewrite"] = ""
        return payload

    def _reconcile_data_logic(
        self,
        *,
        payload: dict[str, object],
        original_payload: dict[str, object],
    ) -> dict[str, object]:
        final_pre = str(payload.get("preprocessing", "")).strip().lower()
        final_window = int(
            payload.get("window_size", original_payload.get("window_size", 128))
        )
        original_pre = str(original_payload.get("preprocessing", "")).strip().lower()
        original_window = int(original_payload.get("window_size", final_window))
        if (final_pre, final_window) == (original_pre, original_window):
            return payload
        function_rewrite = str(payload.get("function_rewrite", ""))
        helper_code = str(payload.get("helper_code", ""))
        logic_expression = str(payload.get("logic_expression", ""))
        combined = " ".join(
            [function_rewrite.lower(), helper_code.lower(), logic_expression.lower()]
        )
        contradictory_pre = (
            original_pre and original_pre in combined and final_pre not in combined
        )
        contradictory_window = (
            str(original_window) in combined and str(final_window) not in combined
        )
        if contradictory_pre or contradictory_window:
            payload["logic_expression"] = ""
            payload["helper_code"] = ""
            payload["function_rewrite"] = ""
        return payload

    def _logic_note(self, payload: dict[str, object]) -> str:
        notes: list[str] = []
        if str(payload.get("function_rewrite", "")).strip():
            notes.append("uses bounded function rewrite")
        elif str(payload.get("helper_code", "")).strip():
            notes.append("uses owned helper code")
        elif str(payload.get("logic_expression", "")).strip():
            notes.append("uses logic expression")
        if not notes:
            return ""
        return " " + "; ".join(notes) + "."

    def _extract_rationale_tags(self, rationale: str) -> list[str]:
        return [match.group(0) for match in re.finditer(r"\[[^\]]+\]", rationale)]

    def _append_tags(self, tags: list[str]) -> str:
        if not tags:
            return ""
        unique_tags = list(dict.fromkeys(tag.strip() for tag in tags if tag.strip()))
        return " " + " ".join(unique_tags)


def build_planner(
    mission_brief: str | None = None,
    mission_source: str | None = None,
) -> OpenAIPlanner | HeuristicPlanner:
    load_local_env()
    backend = os.getenv("AUTOHAR_LLM_BACKEND", "openai").strip().lower()
    allow_fallback = (
        os.getenv("AUTOHAR_ALLOW_LLM_FALLBACK", "true").strip().lower() == "true"
    )

    if backend == "opencode":
        planner = OpenAIPlanner(
            model=os.getenv("AUTOHAR_OPENCODE_MODEL", ""),
            planner_name="opencode_multiagent",
            api_mode="opencode",
            mission_brief=mission_brief,
            mission_source=mission_source,
        )
        if not planner.model:
            planner.model = planner._default_opencode_model()
        return planner

    if backend == "lmstudio":
        base_url = os.getenv("AUTOHAR_LLM_BASE_URL", "http://127.0.0.1:1234/v1")
        api_key = os.getenv("AUTOHAR_LLM_API_KEY", "lm-studio")
        model = os.getenv("AUTOHAR_LLM_MODEL", "local-model")
        return OpenAIPlanner(
            model=model,
            base_url=base_url,
            api_key=api_key,
            planner_name="lmstudio_chat_multiagent",
            api_mode="chat",
            mission_brief=mission_brief,
            mission_source=mission_source,
        )

    if not os.getenv("OPENAI_API_KEY"):
        if allow_fallback:
            from autohar_researcher.heuristic_planner import HeuristicPlanner

            print(
                "[AutoHAR] Warning: No OpenAI API key found. Using heuristic fallback planner."
            )
            print(
                "[AutoHAR] Options: Set OPENAI_API_KEY or use AUTOHAR_LLM_BACKEND=opencode"
            )
            return HeuristicPlanner(
                mission_brief=mission_brief,
                mission_source=mission_source,
            )
        raise RuntimeError(
            "OPENAI_API_KEY is required for the default OpenAI planner. "
            "Options: Set OPENAI_API_KEY or use AUTOHAR_LLM_BACKEND=opencode (local CLI planner)."
        )

    if OpenAI is None:
        if allow_fallback:
            from autohar_researcher.heuristic_planner import HeuristicPlanner

            print(
                "[AutoHAR] Warning: openai package not installed. Using heuristic fallback planner."
            )
            return HeuristicPlanner(
                mission_brief=mission_brief,
                mission_source=mission_source,
            )
        raise RuntimeError(
            "The openai Python package is required for planning. "
            "Install project dependencies or set AUTOHAR_ALLOW_LLM_FALLBACK=true for heuristic fallback."
        )

    return OpenAIPlanner(
        mission_brief=mission_brief,
        mission_source=mission_source,
    )
