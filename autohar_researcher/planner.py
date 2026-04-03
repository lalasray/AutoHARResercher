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
    name = "openai_responses"

    def __init__(self, model: str | None = None) -> None:
        if OpenAI is None:
            raise RuntimeError("openai package is not installed")
        self.model = model or os.getenv("AUTOHAR_LLM_MODEL", "gpt-5-mini")
        self.client = OpenAI()

    def plan_round(
        self,
        round_id: int,
        best_result: CandidateResult | None,
        history: list[CandidateResult],
        accepted_history: list[CandidateResult],
    ) -> dict[str, object]:
        prompt = self._build_prompt(
            round_id=round_id,
            best_result=best_result,
            history=history,
            accepted_history=accepted_history,
        )
        response = self.client.responses.create(
            model=self.model,
            input=prompt,
            text={
                "format": {
                    "type": "json_schema",
                    "name": "autohar_round_plan",
                    "schema": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "summary": {"type": "string"},
                            "model_agent": {
                                "type": "object",
                                "additionalProperties": False,
                                "properties": {
                                    "model_name": {"type": "string"},
                                    "learning_rate": {"type": "number"},
                                    "rationale": {"type": "string"},
                                },
                                "required": ["model_name", "learning_rate", "rationale"],
                            },
                            "augmentation_agent": {
                                "type": "object",
                                "additionalProperties": False,
                                "properties": {
                                    "augmentation": {"type": "string"},
                                    "strength": {"type": "string"},
                                    "rationale": {"type": "string"},
                                },
                                "required": ["augmentation", "strength", "rationale"],
                            },
                            "data_agent": {
                                "type": "object",
                                "additionalProperties": False,
                                "properties": {
                                    "preprocessing": {"type": "string"},
                                    "window_size": {"type": "integer"},
                                    "rationale": {"type": "string"},
                                },
                                "required": ["preprocessing", "window_size", "rationale"],
                            },
                        },
                        "required": [
                            "summary",
                            "model_agent",
                            "augmentation_agent",
                            "data_agent",
                        ],
                    },
                    "strict": True,
                }
            },
        )
        payload = json.loads(response.output_text)
        hints: dict[str, object] = {
            "model_agent": ModelPlanHint(**payload["model_agent"]),
            "augmentation_agent": AugmentationPlanHint(**payload["augmentation_agent"]),
            "data_agent": DataPlanHint(**payload["data_agent"]),
        }
        return {
            "planner": self.name,
            "model": self.model,
            "summary": payload["summary"],
            "hints": hints,
        }

    def _build_prompt(
        self,
        round_id: int,
        best_result: CandidateResult | None,
        history: list[CandidateResult],
        accepted_history: list[CandidateResult],
    ) -> str:
        recent_trials = [
            {
                "round_id": result.round_id,
                "agent_name": result.agent_name,
                "score": round(result.val_score, 4),
                "config": asdict(result.config),
            }
            for result in history[-12:]
        ]
        accepted = [
            {
                "round_id": result.round_id,
                "agent_name": result.agent_name,
                "score": round(result.val_score, 4),
                "config": asdict(result.config),
            }
            for result in accepted_history[-6:]
        ]
        best_payload = None
        if best_result is not None:
            best_payload = {
                "round_id": best_result.round_id,
                "agent_name": best_result.agent_name,
                "score": round(best_result.val_score, 4),
                "config": asdict(best_result.config),
            }
        return (
            "You are the planning controller for an IMU HAR auto-research system. "
            "Suggest one candidate mutation for each agent. "
            "Be trend-aware, prefer ideas that can beat the accepted baseline, and avoid repeating weak patterns.\n\n"
            f"Round: {round_id}\n"
            f"Best accepted result: {json.dumps(best_payload, indent=2)}\n"
            f"Recent accepted history: {json.dumps(accepted, indent=2)}\n"
            f"Recent trials: {json.dumps(recent_trials, indent=2)}\n\n"
            "Constraints:\n"
            "- model_agent may change only model_name and learning_rate\n"
            "- augmentation_agent may change only augmentation and strength\n"
            "- data_agent may change only preprocessing and window_size\n"
            "- Use valid values already seen in the repo such as models {deepconvlstm, hart, tinierhar, cnn_gru, cnn, tiny_transformer, gru}, "
            "augmentations {none, rotation, time_warp, jitter, mixup, scaling, drift, permutation, magnitude_warp}, "
            "strength {weak, medium, strong}, preprocessing {raw, normalized, normalized+gravity_filter, normalized+magnitude_channels}, "
            "window sizes {64, 96, 128, 192, 256}.\n"
            "- The accepted baseline should only be replaced when a candidate scores better.\n"
            "- Prefer concise rationales grounded in observed trends."
        )


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
