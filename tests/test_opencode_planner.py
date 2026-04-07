"""Tests for opencode_planner module."""

from __future__ import annotations

import pytest

from autohar_researcher.opencode_planner import (
    OpenCodePlanner,
    OpenCodeTaskRunner,
    AGENT_GUIDES,
)
from autohar_researcher.git_agent_bridge import GitAgentBridge


class TestAgentGuides:
    def test_guides_exist(self):
        assert "orchestrator" in AGENT_GUIDES
        assert "model_agent" in AGENT_GUIDES
        assert "augmentation_agent" in AGENT_GUIDES
        assert "data_agent" in AGENT_GUIDES
        assert "judge" in AGENT_GUIDES

    def test_guides_non_empty(self):
        for agent, guide in AGENT_GUIDES.items():
            assert len(guide) > 0
            assert agent in guide


class TestOpenCodePlanner:
    def test_initialization(self, tmp_path):
        bridge = GitAgentBridge(repo_root=tmp_path)
        planner = OpenCodePlanner(repo_root=tmp_path, bridge=bridge)

        assert planner.name == "opencode_git"
        assert planner.repo_root == tmp_path
        assert planner.bridge is bridge

    def test_initialization_default_bridge(self, tmp_path):
        planner = OpenCodePlanner(repo_root=tmp_path)
        assert planner.bridge is not None
        assert planner.bridge.repo_root == tmp_path

    def test_fallback_is_heuristic(self, tmp_path):
        planner = OpenCodePlanner(repo_root=tmp_path)
        assert planner._fallback is not None
        assert planner._fallback.name == "heuristic_fallback"

    def test_summarize_history_empty(self, tmp_path):
        planner = OpenCodePlanner(repo_root=tmp_path)
        summary = planner._summarize_history([])
        assert summary == "No history"

    def test_proposal_to_dict(self, tmp_path):
        from autohar_researcher.types import ModelProposal

        planner = OpenCodePlanner(repo_root=tmp_path)
        proposal = ModelProposal(
            model_name="cnn",
            learning_rate=0.001,
            model_params={"hidden_dim": 32},
            notes="test",
        )

        result = planner._proposal_to_dict(proposal)
        assert isinstance(result, dict)
        assert result["model_name"] == "cnn"
        assert result["learning_rate"] == 0.001


class TestOpenCodeTaskRunner:
    def test_initialization(self, tmp_path):
        bridge = GitAgentBridge(repo_root=tmp_path)
        runner = OpenCodeTaskRunner(bridge=bridge)

        assert runner.bridge is bridge
        assert runner.opencode_cmd == "opencode"

    def test_initialization_custom_cmd(self, tmp_path):
        bridge = GitAgentBridge(repo_root=tmp_path)
        runner = OpenCodeTaskRunner(
            bridge=bridge,
            opencode_cmd="/usr/local/bin/opencode",
        )

        assert runner.opencode_cmd == "/usr/local/bin/opencode"

    def test_build_task_content(self, tmp_path):
        from autohar_researcher.git_agent_bridge import AgentTask

        bridge = GitAgentBridge(repo_root=tmp_path)
        runner = OpenCodeTaskRunner(bridge=bridge)

        task = AgentTask(
            task_id="test_1_abc123",
            agent_name="model_agent",
            round_id=1,
            task_type="propose_model",
            prompt="Propose a model for HAR",
            context={"baseline": "deepconvlstm"},
            created_at="2024-01-01T00:00:00Z",
        )

        content = runner._build_task_content(task)
        assert "Task for model_agent" in content
        assert "Round 1" in content
        assert "Propose a model for HAR" in content
        assert "deepconvlstm" in content

    def test_write_task_file(self, tmp_path):
        from autohar_researcher.git_agent_bridge import AgentTask

        bridge = GitAgentBridge(repo_root=tmp_path)
        runner = OpenCodeTaskRunner(bridge=bridge)

        task = AgentTask(
            task_id="test_1_abc123",
            agent_name="augmentation_agent",
            round_id=2,
            task_type="propose_augmentation",
            prompt="Propose augmentation",
            context={},
            created_at="2024-01-01T00:00:00Z",
        )

        content = "Test content"
        task_file = runner._write_task_file(task, content)

        assert task_file.exists()
        assert task_file.read_text() == content
        assert "augmentation_agent" in str(task_file)
        assert "test_1_abc123_content.md" in str(task_file)
