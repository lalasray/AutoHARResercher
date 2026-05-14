"""Heuristic fallback planner when no LLM is available."""

from __future__ import annotations

import random
from dataclasses import asdict
from typing import TYPE_CHECKING

from autohar_researcher.experiment import (
    available_model_architectures,
    default_model_params,
)
from autohar_researcher.types import (
    AugmentationPlanHint,
    CandidateResult,
    DataPlanHint,
    ModelPlanHint,
    ProposalJudgment,
)

if TYPE_CHECKING:
    from autohar_researcher.datasets import WindowedHARDataset


MODEL_FAMILIES = available_model_architectures()

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
    "channel_dropout",
    "sensor_bias",
    "frequency_mask",
    "imu_to_imu",
    "text_to_imu",
    "video_to_imu",
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


class HeuristicPlanner:
    name = "heuristic_fallback"

    def __init__(
        self,
        dataset: WindowedHARDataset | None = None,
        mission_brief: str | None = None,
        mission_source: str | None = None,
    ) -> None:
        self.dataset = dataset
        self._rng = random.Random()
        self.mission_brief = (mission_brief or "").strip()
        self.mission_source = mission_source

    def plan_round(
        self,
        round_id: int,
        best_result: CandidateResult | None,
        history: list[CandidateResult],
        accepted_history: list[CandidateResult],
    ) -> dict[str, object]:
        if round_id == 1:
            return self._plan_first_round(best_result)
        return self._plan_later_round(round_id, best_result, history, accepted_history)

    def _plan_first_round(
        self, best_result: CandidateResult | None
    ) -> dict[str, object]:
        return {
            "planner": self.name,
            "summary": "First round: probe with diverse strong defaults.",
            "mission_brief": self._mission_payload(),
            "hints": {
                "model_agent": ModelPlanHint(
                    model_name="deepconvlstm",
                    learning_rate=0.0003,
                    model_params=default_model_params("deepconvlstm"),
                    rationale="Start with proven deepconvlstm architecture.",
                ),
                "augmentation_agent": AugmentationPlanHint(
                    augmentation="none",
                    strength="medium",
                    rationale="Establish baseline without augmentation first.",
                ),
                "data_agent": DataPlanHint(
                    preprocessing="raw",
                    window_size=128,
                    rationale="Use standard window size for initial baseline.",
                ),
            },
            "orchestrator_plan": {
                "priority_order": ["model", "augmentation", "data"],
                "exploration_mode": "exploit",
                "combination_bias": "stability",
                "risk_note": "Conservative first round.",
            },
            "agent_memories": {
                "model_agent": {"trial_count": 0, "best_trial": None},
                "augmentation_agent": {"trial_count": 0, "best_trial": None},
                "data_agent": {"trial_count": 0, "best_trial": None},
            },
            "combination_memory": {},
        }

    def _plan_later_round(
        self,
        round_id: int,
        best_result: CandidateResult | None,
        history: list[CandidateResult],
        accepted_history: list[CandidateResult],
    ) -> dict[str, object]:
        recent_scores = self._recent_scores(history)
        best_by_agent = self._best_by_agent(history)

        model_hint = self._heuristic_model_hint(best_result, best_by_agent, round_id)
        aug_hint = self._heuristic_augmentation_hint(
            best_result, best_by_agent, round_id
        )
        data_hint = self._heuristic_data_hint(best_result, best_by_agent, round_id)

        exploration_mode, combination_bias = self._select_strategy(
            accepted_history, recent_scores, round_id
        )

        return {
            "planner": self.name,
            "summary": f"Heuristic round {round_id}: {exploration_mode} mode.",
            "mission_brief": self._mission_payload(),
            "hints": {
                "model_agent": model_hint,
                "augmentation_agent": aug_hint,
                "data_agent": data_hint,
            },
            "orchestrator_plan": {
                "priority_order": ["model", "augmentation", "data"],
                "exploration_mode": exploration_mode,
                "combination_bias": combination_bias,
                "risk_note": "Heuristic planner - using simple exploration rules.",
            },
            "agent_memories": {
                agent: {
                    "trial_count": len([r for r in history if r.agent_name == agent])
                }
                for agent in ["model_agent", "augmentation_agent", "data_agent"]
            },
            "combination_memory": {},
        }

    def _mission_payload(self) -> dict[str, object] | None:
        if not self.mission_brief:
            return None
        return {
            "source": self.mission_source,
            "content": self.mission_brief,
        }

    def _heuristic_model_hint(
        self,
        best_result: CandidateResult | None,
        best_by_agent: dict[str, CandidateResult],
        round_id: int,
    ) -> ModelPlanHint:
        if best_result:
            current_model = best_result.config.model_name
            current_lr = best_result.config.learning_rate
            if round_id % 3 == 0:
                return ModelPlanHint(
                    model_name=current_model,
                    learning_rate=current_lr * 0.8,
                    model_params=dict(best_result.config.model_params),
                    rationale="Fine-tune learning rate.",
                )
            elif round_id % 3 == 1:
                next_model = self._next_model(current_model)
                return ModelPlanHint(
                    model_name=next_model,
                    learning_rate=0.0003,
                    model_params=default_model_params(next_model),
                    rationale=f"Try next model family: {next_model}.",
                )

        return ModelPlanHint(
            model_name="deepconvlstm",
            learning_rate=0.0003,
            model_params=default_model_params("deepconvlstm"),
            rationale="Use proven architecture.",
        )

    def _heuristic_augmentation_hint(
        self,
        best_result: CandidateResult | None,
        best_by_agent: dict[str, CandidateResult],
        round_id: int,
    ) -> AugmentationPlanHint:
        if best_result:
            current_aug = best_result.config.augmentation
            current_strength = best_result.config.augmentation_strength
            if current_aug == "none" and round_id > 2:
                return AugmentationPlanHint(
                    augmentation="rotation",
                    strength="medium",
                    rationale="Try weak augmentation.",
                )
            elif current_aug != "none" and round_id % 2 == 0:
                return AugmentationPlanHint(
                    augmentation="none",
                    strength="medium",
                    rationale="Compare with no augmentation.",
                )
            elif current_aug != "none":
                next_aug = self._next_augmentation(current_aug)
                next_strength = self._cycle_strength(current_strength)
                return AugmentationPlanHint(
                    augmentation=next_aug,
                    strength=next_strength,
                    rationale=f"Try {next_aug} at {next_strength} strength.",
                )

        return AugmentationPlanHint(
            augmentation="none",
            strength="medium",
            rationale="Start without augmentation.",
        )

    def _heuristic_data_hint(
        self,
        best_result: CandidateResult | None,
        best_by_agent: dict[str, CandidateResult],
        round_id: int,
    ) -> DataPlanHint:
        if best_result:
            current_preprocessing = best_result.config.preprocessing
            current_window = best_result.config.window_size
            if round_id % 2 == 0:
                next_window = self._next_window(current_window)
                return DataPlanHint(
                    preprocessing=current_preprocessing,
                    window_size=next_window,
                    rationale=f"Try window size {next_window}.",
                )
            else:
                next_preprocessing = self._next_preprocessing(current_preprocessing)
                return DataPlanHint(
                    preprocessing=next_preprocessing,
                    window_size=current_window,
                    rationale=f"Try {next_preprocessing}.",
                )

        return DataPlanHint(
            preprocessing="raw",
            window_size=128,
            rationale="Use standard preprocessing.",
        )

    def _select_strategy(
        self,
        accepted_history: list[CandidateResult],
        recent_scores: list[float],
        round_id: int,
    ) -> tuple[str, str]:
        if len(accepted_history) >= 3:
            recent_accepts = [r.val_score for r in accepted_history[-3:]]
            if max(recent_accepts) - min(recent_accepts) < 0.01:
                return "explore", "broader"

        if round_id <= 3:
            return "exploit", "stability"

        return "explore", "pairwise"

    def _recent_scores(self, history: list[CandidateResult]) -> list[float]:
        return [r.val_score for r in history[-10:]]

    def _best_by_agent(
        self, history: list[CandidateResult]
    ) -> dict[str, CandidateResult]:
        best: dict[str, CandidateResult] = {}
        for result in history:
            agent = result.agent_name
            if agent not in best or result.val_score > best[agent].val_score:
                best[agent] = result
        return best

    def _next_model(self, current: str) -> str:
        if current in MODEL_FAMILIES:
            idx = MODEL_FAMILIES.index(current)
            return MODEL_FAMILIES[(idx + 1) % len(MODEL_FAMILIES)]
        return "deepconvlstm"

    def _next_augmentation(self, current: str) -> str:
        if current in AUGMENTATIONS:
            idx = AUGMENTATIONS.index(current)
            return AUGMENTATIONS[(idx + 1) % len(AUGMENTATIONS)]
        return "rotation"

    def _next_preprocessing(self, current: str) -> str:
        if current in PREPROCESSING_MODES:
            idx = PREPROCESSING_MODES.index(current)
            return PREPROCESSING_MODES[(idx + 1) % len(PREPROCESSING_MODES)]
        return "normalized"

    def _next_window(self, current: int) -> int:
        if current in WINDOW_SIZES:
            idx = WINDOW_SIZES.index(current)
            return WINDOW_SIZES[(idx + 1) % len(WINDOW_SIZES)]
        return 128

    def _cycle_strength(self, current: str) -> str:
        strengths = ["weak", "medium", "strong"]
        if current in strengths:
            idx = strengths.index(current)
            return strengths[(idx + 1) % len(strengths)]
        return "medium"

    def judge_proposals(
        self,
        round_id: int,
        best_result: CandidateResult | None,
        history: list[CandidateResult],
        accepted_history: list[CandidateResult],
        orchestrator_plan: dict[str, object],
        proposals: dict[str, object],
    ) -> dict[str, ProposalJudgment]:
        judgments: dict[str, ProposalJudgment] = {}
        for agent_name, proposal in proposals.items():
            judgments[agent_name] = ProposalJudgment(
                agent_name=agent_name,
                mutation_size="medium",
                novelty="medium",
                plausibility="high",
                compatibility="high",
                evaluate_now=True,
                rationale="Heuristic planner accepts all proposals.",
            )
        return judgments
