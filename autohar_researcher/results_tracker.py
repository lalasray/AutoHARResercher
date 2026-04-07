from __future__ import annotations

import subprocess
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path


@dataclass
class ExperimentResult:
    commit: str
    val_score: float
    memory_gb: float
    status: str
    description: str
    training_seconds: float = 0.0
    total_seconds: float = 0.0
    peak_vram_mb: float = 0.0
    total_tokens_M: float = 0.0
    num_steps: int = 0
    num_params_M: float = 0.0
    depth: int = 0
    domain: str = ""


@dataclass
class ResultsTracker:
    path: Path
    entries: list[ExperimentResult] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.path = Path(self.path)
        if not self.path.exists():
            self._write_header()

    def _write_header(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            "commit\tval_score\tmemory_gb\tstatus\tdescription\t"
            "training_seconds\ttotal_seconds\tpeak_vram_mb\t"
            "total_tokens_M\tnum_steps\tnum_params_M\tdepth\tdomain\n",
            encoding="utf-8",
        )

    def _short_commit(self, commit: str) -> str:
        return commit[:7] if len(commit) > 7 else commit

    def add(self, result: ExperimentResult) -> None:
        self.entries.append(result)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(
                f"{self._short_commit(result.commit)}\t"
                f"{result.val_score:.6f}\t"
                f"{result.memory_gb:.1f}\t"
                f"{result.status}\t"
                f"{result.description}\t"
                f"{result.training_seconds:.1f}\t"
                f"{result.total_seconds:.1f}\t"
                f"{result.peak_vram_mb:.1f}\t"
                f"{result.total_tokens_M:.1f}\t"
                f"{result.num_steps}\t"
                f"{result.num_params_M:.1f}\t"
                f"{result.depth}\t"
                f"{result.domain}\n"
            )

    def get_latest(self) -> ExperimentResult | None:
        return self.entries[-1] if self.entries else None

    def get_best(self) -> ExperimentResult | None:
        if not self.entries:
            return None
        return min(self.entries, key=lambda r: r.val_score)

    def get_keep_entries(self) -> list[ExperimentResult]:
        return [e for e in self.entries if e.status == "keep"]

    def load_from_file(self) -> None:
        if not self.path.exists():
            return
        self.entries = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if line.startswith("commit") or not line.strip():
                continue
            parts = line.split("\t")
            if len(parts) < 5:
                continue
            try:
                self.entries.append(
                    ExperimentResult(
                        commit=parts[0],
                        val_score=float(parts[1]),
                        memory_gb=float(parts[2]),
                        status=parts[3],
                        description=parts[4],
                        training_seconds=float(parts[5])
                        if len(parts) > 5 and parts[5]
                        else 0.0,
                        total_seconds=float(parts[6])
                        if len(parts) > 6 and parts[6]
                        else 0.0,
                        peak_vram_mb=float(parts[7])
                        if len(parts) > 7 and parts[7]
                        else 0.0,
                        total_tokens_M=float(parts[8])
                        if len(parts) > 8 and parts[8]
                        else 0.0,
                        num_steps=int(parts[9]) if len(parts) > 9 and parts[9] else 0,
                        num_params_M=float(parts[10])
                        if len(parts) > 10 and parts[10]
                        else 0.0,
                        depth=int(parts[11]) if len(parts) > 11 and parts[11] else 0,
                        domain=parts[12] if len(parts) > 12 else "",
                    )
                )
            except (ValueError, IndexError):
                continue


def get_git_commit(repo_root: Path | None = None) -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root or Path.cwd(),
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError:
        return "unknown"


def get_git_branch(repo_root: Path | None = None) -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=repo_root or Path.cwd(),
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError:
        return "unknown"


def create_results_tracker(
    results_path: Path | str | None = None,
    repo_root: Path | None = None,
) -> ResultsTracker:
    if results_path is None:
        repo = repo_root or Path.cwd()
        results_path = repo / "results.tsv"
    return ResultsTracker(Path(results_path))
