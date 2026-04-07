from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from autohar_researcher.git_agent_bridge import AgentTask, GitAgentBridge
from autohar_researcher.types import (
    AugmentationPlanHint,
    CandidateResult,
    DataPlanHint,
    ModelPlanHint,
    ProposalJudgment,
)

from autohar_researcher.heuristic_planner import HeuristicPlanner


AGENT_GUIDES = {
    "orchestrator": """You are the top-level controller for AutoHAR research system.
Your job is to plan the next research round based on historical results.
Coordinate model, augmentation, and data agents to improve HAR model performance.""",
    "model_agent": """You are the ModelAgent for AutoHAR research.
Your job is to propose the best model configuration for the current round.
Focus on model architecture, learning rate, and model-specific parameters.""",
    "augmentation_agent": """You are the AugmentationAgent for AutoHAR research.
Your job is to propose the best data augmentation strategy.
Focus on augmentation type, strength, and parameters.""",
    "data_agent": """You are the DataAgent for AutoHAR research.
Your job is to propose the best data preprocessing configuration.
Focus on preprocessing mode, window size, and data parameters.""",
    "judge": """You are the Judge for AutoHAR research.
Your job is to evaluate proposals and decide which ones to accept.
Consider novelty, plausibility, and compatibility with the baseline.""",
}


class OpenCodePlanner:
    name = "opencode_git"

    def __init__(
        self,
        repo_root: Path | None = None,
        bridge: GitAgentBridge | None = None,
        timeout: float = 300.0,
    ) -> None:
        self.repo_root = repo_root or Path.cwd()
        self.bridge = bridge or GitAgentBridge(repo_root=repo_root)
        self.timeout = timeout
        self._fallback = HeuristicPlanner()

    def plan_round(
        self,
        round_id: int,
        best_result: CandidateResult | None,
        history: list[CandidateResult],
        accepted_history: list[CandidateResult],
    ) -> dict[str, Any]:
        orchestrator_task = self._build_orchestrator_task(
            round_id, best_result, history, accepted_history
        )

        task = self.bridge.submit_task(
            agent_name="orchestrator",
            round_id=round_id,
            task_type="plan_round",
            prompt=self._build_orchestrator_prompt(
                round_id, best_result, history, accepted_history
            ),
            context={
                "history_summary": self._summarize_history(history),
                "accepted_summary": self._summarize_history(accepted_history),
            },
        )

        result = self.bridge.wait_for_result(task, branch="main")
        if result.status == "timeout":
            return self._fallback.plan_round(
                round_id, best_result, history, accepted_history
            )

        return self._parse_orchestrator_result(result.result)

    def _build_orchestrator_task(
        self,
        round_id: int,
        best_result: CandidateResult | None,
        history: list[CandidateResult],
        accepted_history: list[CandidateResult],
    ) -> dict[str, Any]:
        return {
            "round_id": round_id,
            "best_result": self._result_to_dict(best_result) if best_result else None,
            "history_count": len(history),
            "accepted_count": len(accepted_history),
        }

    def _build_orchestrator_prompt(
        self,
        round_id: int,
        best_result: CandidateResult | None,
        history: list[CandidateResult],
        accepted_history: list[CandidateResult],
    ) -> str:
        history_lines = []
        for r in history[-10:]:
            history_lines.append(
                f"- Round {r.round_id}: {r.agent_name} -> F1={r.val_score:.4f} (accepted={r.accepted})"
            )

        accepted_lines = []
        for r in accepted_history[-5:]:
            accepted_lines.append(
                f"- Round {r.round_id}: {r.agent_name} -> F1={r.val_score:.4f}"
            )

        best_info = ""
        if best_result:
            best_info = f"""
Current best: {best_result.config.model_name} with F1={best_result.val_score:.4f}
- Preprocessing: {best_result.config.preprocessing}
- Window: {best_result.config.window_size}
- Augmentation: {best_result.config.augmentation} ({best_result.config.augmentation_strength})
- Learning Rate: {best_result.config.learning_rate}
"""

        return f"""You are planning round {round_id} of AutoHAR research.

## Your Role
{AGENT_GUIDES["orchestrator"]}

## Current State
{best_info}

## Recent History
{chr(10).join(history_lines) if history_lines else "No history yet"}

## Accepted Results
{chr(10).join(accepted_lines) if accepted_lines else "No accepted results yet"}

## Task
Analyze the history and propose:
1. Priority order for agents (model, augmentation, data)
2. Exploration mode (exploit or explore)
3. Risk level for this round
4. Any specific directions for each agent

Return a JSON plan in this format:
```json
{{
  "priority_order": ["model", "augmentation", "data"],
  "exploration_mode": "exploit",
  "combination_bias": "stability",
  "risk_level": "low",
  "model_direction": "Try attention-based architectures",
  "augmentation_direction": "Focus on rotation augmentation",
  "data_direction": "Use normalized preprocessing"
}}
```"""

    def _parse_orchestrator_result(self, result: str) -> dict[str, Any]:
        try:
            for line in result.split("\n"):
                if line.strip().startswith("{"):
                    start = result.index(line.strip())
                    for i in range(len(result) - 1, start - 1, -1):
                        if result[i] == "}":
                            json_str = result[start : i + 1]
                            data = json.loads(json_str)
                            return self._normalize_orchestrator_plan(data)
            return self._fallback.plan_round(1, None, [], [])
        except (json.JSONDecodeError, ValueError):
            return self._fallback.plan_round(1, None, [], [])

    def _normalize_orchestrator_plan(self, data: dict[str, Any]) -> dict[str, Any]:
        return {
            "planner": self.name,
            "summary": data.get("model_direction", "OpenCode planned round"),
            "hints": {
                "model_agent": ModelPlanHint(
                    model_name=data.get("model_name", "deepconvlstm"),
                    learning_rate=data.get("learning_rate", 0.0003),
                    model_params=data.get("model_params", {}),
                    rationale=data.get("model_direction", "OpenCode suggestion"),
                ),
                "augmentation_agent": AugmentationPlanHint(
                    augmentation=data.get("augmentation", "none"),
                    strength=data.get("augmentation_strength", "medium"),
                    rationale=data.get("augmentation_direction", "OpenCode suggestion"),
                ),
                "data_agent": DataPlanHint(
                    preprocessing=data.get("preprocessing", "raw"),
                    window_size=data.get("window_size", 128),
                    rationale=data.get("data_direction", "OpenCode suggestion"),
                ),
            },
            "orchestrator_plan": {
                "priority_order": data.get(
                    "priority_order", ["model", "augmentation", "data"]
                ),
                "exploration_mode": data.get("exploration_mode", "exploit"),
                "combination_bias": data.get("combination_bias", "stability"),
                "risk_level": data.get("risk_level", "medium"),
            },
            "agent_memories": {
                "model_agent": {"trial_count": 0, "best_trial": None},
                "augmentation_agent": {"trial_count": 0, "best_trial": None},
                "data_agent": {"trial_count": 0, "best_trial": None},
            },
            "combination_memory": {},
        }

    def judge_proposals(
        self,
        round_id: int,
        best_result: CandidateResult | None,
        history: list[CandidateResult],
        accepted_history: list[CandidateResult],
        orchestrator_plan: dict[str, Any],
        proposals: dict[str, object],
    ) -> dict[str, ProposalJudgment]:
        judgments: dict[str, ProposalJudgment] = {}

        for agent_name, proposal in proposals.items():
            task = self.bridge.submit_task(
                agent_name="judge",
                round_id=round_id,
                task_type="judge_proposal",
                prompt=self._build_judge_prompt(
                    agent_name, proposal, best_result, history
                ),
                context={"proposal": self._proposal_to_dict(proposal)},
            )

            result = self.bridge.wait_for_result(task, branch="main")
            if result.status == "timeout":
                judgments[agent_name] = self._fallback_judgment(agent_name)
            else:
                judgments[agent_name] = self._parse_judgment(agent_name, result.result)

        return judgments

    def _build_judge_prompt(
        self,
        agent_name: str,
        proposal: object,
        best_result: CandidateResult | None,
        history: list[CandidateResult],
    ) -> str:
        proposal_str = json.dumps(self._proposal_to_dict(proposal), indent=2)
        best_info = ""
        if best_result:
            best_info = f"""
Current best F1: {best_result.val_score:.4f}
Best config: {best_result.config.model_name}, {best_result.config.preprocessing}, {best_result.config.augmentation}
"""

        return f"""You are judging a proposal from {agent_name}.

## Your Role
{AGENT_GUIDES["judge"]}

## Proposal to Judge
```json
{proposal_str}
```

{best_info}

## Recent Performance
Consider the recent history to assess novelty and plausibility.

## Task
Evaluate this proposal and return a judgment:
```json
{{
  "mutation_size": "small/medium/large",
  "novelty": "low/medium/high",
  "plausibility": "low/medium/high",
  "compatibility": "low/medium/high",
  "evaluate_now": true/false,
  "rationale": "..."
}}
```"""

    def _parse_judgment(self, agent_name: str, result: str) -> ProposalJudgment:
        try:
            for line in result.split("\n"):
                if line.strip().startswith("{"):
                    start = result.index(line.strip())
                    for i in range(len(result) - 1, start - 1, -1):
                        if result[i] == "}":
                            json_str = result[start : i + 1]
                            data = json.loads(json_str)
                            return ProposalJudgment(
                                agent_name=agent_name,
                                mutation_size=data.get("mutation_size", "medium"),
                                novelty=data.get("novelty", "medium"),
                                plausibility=data.get("plausibility", "medium"),
                                compatibility=data.get("compatibility", "medium"),
                                evaluate_now=data.get("evaluate_now", True),
                                rationale=data.get("rationale", ""),
                            )
            return self._fallback_judgment(agent_name)
        except (json.JSONDecodeError, ValueError):
            return self._fallback_judgment(agent_name)

    def _fallback_judgment(self, agent_name: str) -> ProposalJudgment:
        return ProposalJudgment(
            agent_name=agent_name,
            mutation_size="medium",
            novelty="medium",
            plausibility="high",
            compatibility="high",
            evaluate_now=True,
            rationale="Fallback judgment - OpenCode timed out",
        )

    def _summarize_history(self, history: list[CandidateResult]) -> str:
        if not history:
            return "No history"
        lines = []
        for r in history[-5:]:
            lines.append(f"{r.agent_name}: F1={r.val_score:.4f}")
        return "\n".join(lines)

    def _result_to_dict(self, result: CandidateResult | None) -> dict[str, Any] | None:
        if result is None:
            return None
        return {
            "agent_name": result.agent_name,
            "round_id": result.round_id,
            "val_score": result.val_score,
            "config": self._proposal_to_dict(result.config),
        }

    def _proposal_to_dict(self, proposal: object) -> dict[str, Any]:
        if hasattr(proposal, "__dict__"):
            return vars(proposal)
        if isinstance(proposal, dict):
            return proposal
        return {"value": str(proposal)}


def build_opencode_planner(repo_root: Path | None = None) -> OpenCodePlanner:
    return OpenCodePlanner(repo_root=repo_root)
