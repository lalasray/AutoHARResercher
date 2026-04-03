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
        self.guides = self._load_guides()

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
        trend_summary = self._trend_summary(history, accepted_history)
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
            f"Orchestrator guide:\n{self.guides['orchestrator']}\n\n"
            f"ModelAgent guide:\n{self.guides['model_agent']}\n\n"
            f"AugmentationAgent guide:\n{self.guides['augmentation_agent']}\n\n"
            f"DataAgent guide:\n{self.guides['data_agent']}\n\n"
            f"Round: {round_id}\n"
            f"Best accepted result: {json.dumps(best_payload, indent=2)}\n"
            f"Trend summary: {json.dumps(trend_summary, indent=2)}\n"
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
            "- Prefer concise rationales grounded in observed trends.\n"
            "- If a category has been winning repeatedly, bias toward small exploitative mutations around it.\n"
            "- If a category has been losing repeatedly, either recommend `none` / stability or propose a different focused experiment.\n"
        )

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
