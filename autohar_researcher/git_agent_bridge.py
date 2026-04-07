from __future__ import annotations

import json
import os
import subprocess
import time
import uuid
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any

from autohar_researcher.types import CandidateResult


@dataclass
class AgentTask:
    task_id: str
    agent_name: str
    round_id: int
    task_type: str
    prompt: str
    context: dict[str, Any]
    created_at: str
    status: str = "pending"
    result: str = ""
    completed_at: str = ""


class GitAgentBridge:
    def __init__(
        self,
        repo_root: Path | None = None,
        tasks_dir: str = "research/tasks",
        results_dir: str = "research/results",
        poll_interval: float = 5.0,
        timeout: float = 300.0,
    ) -> None:
        self.repo_root = repo_root or Path.cwd()
        self.tasks_dir = self.repo_root / tasks_dir
        self.results_dir = self.repo_root / results_dir
        self.poll_interval = poll_interval
        self.timeout = timeout

    def submit_task(
        self,
        agent_name: str,
        round_id: int,
        task_type: str,
        prompt: str,
        context: dict[str, Any],
        branch: str | None = None,
    ) -> AgentTask:
        task_id = f"{agent_name}_{round_id}_{uuid.uuid4().hex[:8]}"
        task = AgentTask(
            task_id=task_id,
            agent_name=agent_name,
            round_id=round_id,
            task_type=task_type,
            prompt=prompt,
            context=context,
            created_at=datetime.utcnow().isoformat(),
            status="pending",
        )

        task_dir = self.tasks_dir / agent_name
        task_dir.mkdir(parents=True, exist_ok=True)
        task_file = task_dir / f"{task_id}.json"
        task_file.write_text(
            json.dumps(asdict(task), indent=2) + "\n", encoding="utf-8"
        )

        if branch:
            self._git_add_commit_push(
                paths=[task_file],
                message=f"task: {agent_name} round {round_id} - {task_type}",
                branch=branch,
            )
        else:
            self._git_add_commit(
                paths=[task_file],
                message=f"task: {agent_name} round {round_id} - {task_type}",
            )

        return task

    def wait_for_result(
        self,
        task: AgentTask,
        branch: str | None = None,
    ) -> AgentTask:
        start_time = time.time()
        result_file = self.results_dir / task.agent_name / f"{task.task_id}.json"

        while time.time() - start_time < self.timeout:
            if result_file.exists():
                try:
                    result_data = json.loads(result_file.read_text(encoding="utf-8"))
                    task.status = result_data.get("status", "completed")
                    task.result = result_data.get("result", "")
                    task.completed_at = datetime.utcnow().isoformat()
                    return task
                except (json.JSONDecodeError, KeyError):
                    pass

            self._git_pull(cwd=self.repo_root)
            time.sleep(self.poll_interval)

        task.status = "timeout"
        task.completed_at = datetime.utcnow().isoformat()
        return task

    def write_result(
        self,
        task: AgentTask,
        result: str,
        branch: str | None = None,
    ) -> None:
        task.status = "completed"
        task.result = result
        task.completed_at = datetime.utcnow().isoformat()

        result_dir = self.results_dir / task.agent_name
        result_dir.mkdir(parents=True, exist_ok=True)
        result_file = result_dir / f"{task.task_id}.json"
        result_file.write_text(
            json.dumps(asdict(task), indent=2) + "\n", encoding="utf-8"
        )

        if branch:
            self._git_add_commit_push(
                paths=[result_file],
                message=f"result: {task.agent_name} {task.task_id}",
                branch=branch,
            )
        else:
            self._git_add_commit(
                paths=[result_file],
                message=f"result: {task.agent_name} {task.task_id}",
            )

    def get_pending_tasks(self, agent_name: str) -> list[AgentTask]:
        task_dir = self.tasks_dir / agent_name
        if not task_dir.exists():
            return []

        tasks = []
        for task_file in task_dir.glob("*.json"):
            try:
                data = json.loads(task_file.read_text(encoding="utf-8"))
                if data.get("status") == "pending":
                    tasks.append(AgentTask(**data))
            except (json.JSONDecodeError, TypeError):
                pass

        return sorted(tasks, key=lambda t: t.created_at)

    def get_task_result(self, task: AgentTask) -> AgentTask | None:
        result_file = self.results_dir / task.agent_name / f"{task.task_id}.json"
        if not result_file.exists():
            return None

        try:
            data = json.loads(result_file.read_text(encoding="utf-8"))
            return AgentTask(**data)
        except (json.JSONDecodeError, TypeError):
            return None

    def _git_add_commit(
        self,
        paths: list[Path],
        message: str,
    ) -> bool:
        try:
            for path in paths:
                self.run_git(
                    ["add", str(path.relative_to(self.repo_root))], cwd=self.repo_root
                )
            self.run_git(["commit", "-m", message], cwd=self.repo_root)
            return True
        except subprocess.CalledProcessError:
            return False

    def _git_add_commit_push(
        self,
        paths: list[Path],
        message: str,
        branch: str,
    ) -> bool:
        try:
            self.run_git(["checkout", branch], cwd=self.repo_root)
            for path in paths:
                self.run_git(
                    ["add", str(path.relative_to(self.repo_root))], cwd=self.repo_root
                )
            self.run_git(["commit", "-m", message], cwd=self.repo_root)
            if self._has_remote():
                self.run_git(["push", "-u", "origin", branch], cwd=self.repo_root)
            return True
        except subprocess.CalledProcessError:
            return False

    def _git_pull(self, cwd: Path | None = None) -> None:
        try:
            self.run_git(["pull", "--rebase"], cwd=cwd or self.repo_root)
        except subprocess.CalledProcessError:
            pass

    def _has_remote(self) -> bool:
        try:
            result = self.run_git(["remote", "get-url", "origin"], check=False)
            return result.returncode == 0 and bool(result.stdout.strip())
        except subprocess.CalledProcessError:
            return False

    def run_git(
        self,
        args: list[str],
        cwd: Path | None = None,
        check: bool = True,
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["git", *args],
            cwd=cwd or self.repo_root,
            check=check,
            text=True,
            capture_output=True,
        )
