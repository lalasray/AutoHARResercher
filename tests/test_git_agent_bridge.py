"""Tests for git_agent_bridge module."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import pytest

from autohar_researcher.git_agent_bridge import AgentTask, GitAgentBridge


class TestAgentTask:
    def test_task_creation(self):
        task = AgentTask(
            task_id="test_1_abc123",
            agent_name="model_agent",
            round_id=1,
            task_type="propose_model",
            prompt="Propose a model",
            context={"current_model": "deepconvlstm"},
            created_at="2024-01-01T00:00:00Z",
        )
        assert task.task_id == "test_1_abc123"
        assert task.agent_name == "model_agent"
        assert task.round_id == 1
        assert task.status == "pending"
        assert task.result == ""

    def test_task_with_result(self):
        task = AgentTask(
            task_id="test_1_abc123",
            agent_name="model_agent",
            round_id=1,
            task_type="propose_model",
            prompt="Propose a model",
            context={},
            created_at="2024-01-01T00:00:00Z",
            status="completed",
            result='{"model": "cnn"}',
            completed_at="2024-01-01T00:01:00Z",
        )
        assert task.status == "completed"
        assert task.result == '{"model": "cnn"}'
        assert task.completed_at == "2024-01-01T00:01:00Z"


class TestGitAgentBridge:
    def test_initialization(self, tmp_path):
        bridge = GitAgentBridge(repo_root=tmp_path)
        assert bridge.repo_root == tmp_path
        assert bridge.tasks_dir == tmp_path / "research" / "tasks"
        assert bridge.results_dir == tmp_path / "research" / "results"

    def test_initialization_custom_dirs(self, tmp_path):
        bridge = GitAgentBridge(
            repo_root=tmp_path,
            tasks_dir="tasks",
            results_dir="results",
        )
        assert bridge.tasks_dir == tmp_path / "tasks"
        assert bridge.results_dir == tmp_path / "results"

    def test_submit_task(self, tmp_path):
        bridge = GitAgentBridge(repo_root=tmp_path, timeout=5.0)
        task = bridge.submit_task(
            agent_name="model_agent",
            round_id=1,
            task_type="propose_model",
            prompt="Propose the best model for HAR",
            context={"baseline": "deepconvlstm"},
        )

        assert task.agent_name == "model_agent"
        assert task.round_id == 1
        assert task.task_type == "propose_model"
        assert task.status == "pending"
        assert task.task_id.startswith("model_agent_1_")

        task_file = bridge.tasks_dir / "model_agent" / f"{task.task_id}.json"
        assert task_file.exists()

        with open(task_file) as f:
            data = json.load(f)
        assert data["agent_name"] == "model_agent"
        assert data["round_id"] == 1

    def test_write_result(self, tmp_path):
        bridge = GitAgentBridge(repo_root=tmp_path)
        task = bridge.submit_task(
            agent_name="augmentation_agent",
            round_id=2,
            task_type="propose_augmentation",
            prompt="Propose augmentation",
            context={},
        )

        result = '{"augmentation": "rotation", "strength": "medium"}'
        bridge.write_result(task, result)

        result_file = bridge.results_dir / "augmentation_agent" / f"{task.task_id}.json"
        assert result_file.exists()

        with open(result_file) as f:
            data = json.load(f)
        assert data["status"] == "completed"
        assert data["result"] == result

    def test_get_pending_tasks(self, tmp_path):
        bridge = GitAgentBridge(repo_root=tmp_path)

        task1 = bridge.submit_task(
            agent_name="model_agent",
            round_id=1,
            task_type="propose",
            prompt="Task 1",
            context={},
        )
        task2 = bridge.submit_task(
            agent_name="model_agent",
            round_id=2,
            task_type="propose",
            prompt="Task 2",
            context={},
        )

        pending = bridge.get_pending_tasks("model_agent")
        assert len(pending) == 2
        assert all(t.status == "pending" for t in pending)

    def test_get_pending_tasks_none(self, tmp_path):
        bridge = GitAgentBridge(repo_root=tmp_path)
        pending = bridge.get_pending_tasks("nonexistent_agent")
        assert pending == []

    def test_get_task_result_not_exists(self, tmp_path):
        bridge = GitAgentBridge(repo_root=tmp_path)
        task = AgentTask(
            task_id="test_nonexistent",
            agent_name="model_agent",
            round_id=1,
            task_type="test",
            prompt="test",
            context={},
            created_at="2024-01-01T00:00:00Z",
        )
        result = bridge.get_task_result(task)
        assert result is None

    def test_get_task_result_exists(self, tmp_path):
        bridge = GitAgentBridge(repo_root=tmp_path)
        task = bridge.submit_task(
            agent_name="data_agent",
            round_id=1,
            task_type="test",
            prompt="test",
            context={},
        )

        bridge.write_result(task, '{"preprocessing": "normalized"}')

        result = bridge.get_task_result(task)
        assert result is not None
        assert result.status == "completed"
        assert result.result == '{"preprocessing": "normalized"}'


class TestGitOperations:
    def test_run_git(self, tmp_path):
        bridge = GitAgentBridge(repo_root=tmp_path)

        git_dir = tmp_path / ".git"
        git_dir.mkdir()

        result = bridge.run_git(["status"], cwd=tmp_path, check=False)
        assert result.returncode == 0
        assert "On branch" in result.stdout or "HEAD" in result.stdout

    def test_has_remote_no_remote(self, tmp_path):
        bridge = GitAgentBridge(repo_root=tmp_path)
        assert bridge._has_remote() is False

    def test_git_add_commit(self, tmp_path):
        bridge = GitAgentBridge(repo_root=tmp_path)

        test_file = tmp_path / "test.txt"
        test_file.write_text("test content")

        result = bridge._git_add_commit(
            paths=[test_file],
            message="Test commit",
        )
        assert result is True

        status = bridge.run_git(["status", "--porcelain"], cwd=tmp_path)
        assert test_file.name not in status.stdout
