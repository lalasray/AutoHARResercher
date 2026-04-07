"""Tests for types module."""

from __future__ import annotations

import pytest

from autohar_researcher.types import (
    CandidateConfig,
    CandidateResult,
    ModelProposal,
    AugmentationProposal,
    DataProposal,
    AgentWorkspace,
    GitAction,
    RunOutputs,
    ModelPlanHint,
    AugmentationPlanHint,
    DataPlanHint,
    ProposalJudgment,
)


class TestCandidateConfig:
    def test_creation(self):
        config = CandidateConfig(
            model_name="deepconvlstm",
            preprocessing="raw",
            augmentation="none",
            augmentation_strength="medium",
            learning_rate=0.001,
            window_size=128,
        )
        assert config.model_name == "deepconvlstm"
        assert config.preprocessing == "raw"
        assert config.augmentation == "none"
        assert config.learning_rate == 0.001
        assert config.window_size == 128
        assert config.model_params == {}
        assert config.notes == ""

    def test_with_params(self):
        config = CandidateConfig(
            model_name="deepconvlstm",
            preprocessing="normalized",
            augmentation="rotation",
            augmentation_strength="weak",
            learning_rate=0.0003,
            window_size=64,
            model_params={"hidden_dim": 64, "depth": 2},
            augmentation_params={"angle_deg": 5.0},
            notes="test config",
        )
        assert config.model_params == {"hidden_dim": 64, "depth": 2}
        assert config.augmentation_params == {"angle_deg": 5.0}
        assert config.notes == "test config"


class TestProposals:
    def test_model_proposal(self):
        proposal = ModelProposal(
            model_name="cnn_gru",
            learning_rate=0.0005,
            model_params={"hidden_dim": 48},
            notes="test proposal",
        )
        assert proposal.model_name == "cnn_gru"
        assert proposal.learning_rate == 0.0005
        assert proposal.logic_expression == ""

    def test_augmentation_proposal(self):
        proposal = AugmentationProposal(
            augmentation="jitter",
            strength="strong",
            notes="test augmentation",
        )
        assert proposal.augmentation == "jitter"
        assert proposal.strength == "strong"

    def test_data_proposal(self):
        proposal = DataProposal(
            preprocessing="normalized+gravity_filter",
            window_size=192,
            notes="test data",
        )
        assert proposal.preprocessing == "normalized+gravity_filter"
        assert proposal.window_size == 192


class TestPlanHints:
    def test_model_plan_hint(self):
        hint = ModelPlanHint(
            model_name="deepconvlstmattention",
            learning_rate=0.0002,
            model_params={"attention_mix": 0.8},
            rationale="use attention",
        )
        assert hint.rationale == "use attention"
        assert hint.logic_expression == ""

    def test_augmentation_plan_hint(self):
        hint = AugmentationPlanHint(
            augmentation="rotation",
            strength="medium",
            rationale="improve robustness",
        )
        assert hint.strength == "medium"

    def test_data_plan_hint(self):
        hint = DataPlanHint(
            preprocessing="normalized+magnitude_channels",
            window_size=128,
            rationale="add magnitude features",
        )
        assert hint.window_size == 128


class TestProposalJudgment:
    def test_basic_judgment(self):
        judgment = ProposalJudgment(
            agent_name="model_agent",
            mutation_size="medium",
            novelty="high",
            plausibility="high",
            compatibility="medium",
            evaluate_now=True,
            rationale="promising proposal",
        )
        assert judgment.evaluate_now is True
        assert judgment.mutation_size == "medium"

    def test_reject_judgment(self):
        judgment = ProposalJudgment(
            agent_name="augmentation_agent",
            mutation_size="large",
            novelty="low",
            plausibility="low",
            compatibility="low",
            evaluate_now=False,
            rationale="unsafe proposal",
        )
        assert judgment.evaluate_now is False


class TestAgentWorkspace:
    def test_workspace(self):
        workspace = AgentWorkspace(
            agent_name="model_agent",
            branch_name="agent-model-main",
            worktree_path="/worktrees/model_agent",
        )
        assert workspace.agent_name == "model_agent"
        assert workspace.branch_name == "agent-model-main"


class TestGitAction:
    def test_action(self):
        action = GitAction(
            label="Create branch",
            command="git branch test main",
        )
        assert action.label == "Create branch"
        assert "git branch" in action.command


class TestRunOutputs:
    def test_outputs(self):
        outputs = RunOutputs(
            trial_history_csv="artifacts/trial_history.csv",
            trial_history_svg="artifacts/trial_history.svg",
            accepted_history_csv="artifacts/accepted_history.csv",
            accepted_history_svg="artifacts/accepted_history.svg",
        )
        assert outputs.trial_history_csv.endswith(".csv")
        assert outputs.trial_history_svg.endswith(".svg")
