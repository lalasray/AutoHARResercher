from __future__ import annotations

import difflib
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from autohar_researcher.types import FilePatch


@dataclass
class PatchResult:
    file_path: str
    success: bool
    mode: str
    error: str | None = None
    diff: str = ""


def _validate_path(file_path: str, worktree_root: Path) -> Path | None:
    target = (worktree_root / file_path).resolve()
    root = worktree_root.resolve()
    try:
        target.relative_to(root)
    except ValueError:
        return None
    return target


def _apply_full_rewrite(target: Path, content: str, worktree_root: Path) -> PatchResult:
    target.parent.mkdir(parents=True, exist_ok=True)
    old_content = ""
    if target.exists():
        old_content = target.read_text(encoding="utf-8")
    target.write_text(content, encoding="utf-8")
    diff = "".join(
        difflib.unified_diff(
            old_content.splitlines(keepends=True),
            content.splitlines(keepends=True),
            fromfile=f"a/{target.relative_to(worktree_root)}",
            tofile=f"b/{target.relative_to(worktree_root)}",
        )
    )
    return PatchResult(
        file_path=str(target.relative_to(worktree_root)),
        success=True,
        mode="full_rewrite",
        diff=diff,
    )


def _apply_append(target: Path, content: str, worktree_root: Path) -> PatchResult:
    target.parent.mkdir(parents=True, exist_ok=True)
    old_content = ""
    if target.exists():
        old_content = target.read_text(encoding="utf-8")
    new_content = old_content + "\n" + content if old_content else content
    target.write_text(new_content, encoding="utf-8")
    diff = "".join(
        difflib.unified_diff(
            old_content.splitlines(keepends=True),
            new_content.splitlines(keepends=True),
            fromfile=f"a/{target.relative_to(worktree_root)}",
            tofile=f"b/{target.relative_to(worktree_root)}",
        )
    )
    return PatchResult(
        file_path=str(target.relative_to(worktree_root)),
        success=True,
        mode="append",
        diff=diff,
    )


def _apply_diff_patch(target: Path, content: str, worktree_root: Path) -> PatchResult:
    if not target.exists():
        return PatchResult(
            file_path=str(target.relative_to(worktree_root)),
            success=False,
            mode="diff",
            error="Cannot apply diff: target file does not exist",
        )
    old_content = target.read_text(encoding="utf-8")
    old_lines = old_content.splitlines()
    new_lines = _apply_unified_diff(old_lines, content)
    if new_lines is None:
        return PatchResult(
            file_path=str(target.relative_to(worktree_root)),
            success=False,
            mode="diff",
            error="Failed to apply unified diff patch",
        )
    new_content = "\n".join(new_lines) + ("\n" if old_content.endswith("\n") else "")
    target.write_text(new_content, encoding="utf-8")
    diff = "".join(
        difflib.unified_diff(
            old_content.splitlines(keepends=True),
            new_content.splitlines(keepends=True),
            fromfile=f"a/{target.relative_to(worktree_root)}",
            tofile=f"b/{target.relative_to(worktree_root)}",
        )
    )
    return PatchResult(
        file_path=str(target.relative_to(worktree_root)),
        success=True,
        mode="diff",
        diff=diff,
    )


def _apply_unified_diff(old_lines: list[str], patch_text: str) -> list[str] | None:
    try:
        result: list[str] = list(old_lines)
        current_hunk: dict[str, Any] | None = None
        patch_lines = patch_text.splitlines()

        i = 0
        while i < len(patch_lines):
            line = patch_lines[i]
            if line.startswith("--- ") or line.startswith("+++ "):
                i += 1
                continue
            if line.startswith("@@ "):
                parts = line.split("@@")
                if len(parts) >= 2:
                    hunk_header = parts[1].strip()
                    old_info = hunk_header.split(" ")[0]
                    if old_info.startswith("-"):
                        start = int(old_info[1:].split(",")[0])
                        current_hunk = {"start": start, "deletes": [], "inserts": []}
            elif current_hunk is not None:
                if line.startswith("-"):
                    current_hunk["deletes"].append(line[1:])
                elif line.startswith("+"):
                    current_hunk["inserts"].append(line[1:])
                elif line.startswith(" ") or line == "":
                    if current_hunk["deletes"] or current_hunk["inserts"]:
                        _apply_hunk(result, current_hunk)
                        current_hunk = None
            i += 1

        if current_hunk is not None and (
            current_hunk["deletes"] or current_hunk["inserts"]
        ):
            _apply_hunk(result, current_hunk)

        return result
    except Exception:
        return None


def _apply_hunk(lines: list[str], hunk: dict[str, Any]) -> None:
    start = hunk["start"] - 1
    deletes = hunk["deletes"]
    inserts = hunk["inserts"]

    if start < 0 or start > len(lines):
        return

    if not deletes:
        for idx, ins in enumerate(inserts):
            lines.insert(start + idx, ins)
        return

    match_idx = -1
    for i in range(start, min(start + 3, len(lines) - len(deletes) + 1)):
        if lines[i : i + len(deletes)] == deletes:
            match_idx = i
            break

    if match_idx == -1:
        for i in range(start, min(start + 10, len(lines) - len(deletes) + 1)):
            if all(
                lines[i + j].strip() == deletes[j].strip()
                for j in range(len(deletes))
                if i + j < len(lines)
            ):
                match_idx = i
                break

    if match_idx == -1:
        for idx, ins in enumerate(inserts):
            lines.insert(start + idx, ins)
        return

    for _ in range(len(deletes)):
        lines.pop(match_idx)
    for idx, ins in enumerate(inserts):
        lines.insert(match_idx + idx, ins)


def apply_file_patches(
    patches: list[FilePatch],
    worktree_root: Path,
) -> list[PatchResult]:
    results: list[PatchResult] = []
    for patch in patches:
        target = _validate_path(patch.file_path, worktree_root)
        if target is None:
            results.append(
                PatchResult(
                    file_path=patch.file_path,
                    success=False,
                    mode=patch.patch_mode,
                    error="Path validation failed: file is outside worktree root",
                )
            )
            continue

        try:
            mode = patch.patch_mode.lower()
            if mode == "full_rewrite":
                result = _apply_full_rewrite(target, patch.content, worktree_root)
            elif mode == "append":
                result = _apply_append(target, patch.content, worktree_root)
            elif mode == "diff":
                result = _apply_diff_patch(target, patch.content, worktree_root)
            else:
                result = _apply_full_rewrite(target, patch.content, worktree_root)
        except Exception as exc:
            result = PatchResult(
                file_path=patch.file_path,
                success=False,
                mode=patch.patch_mode,
                error=f"{type(exc).__name__}: {exc}",
            )
        results.append(result)
    return results


def collect_all_file_patches(
    proposals: list[object],
) -> list[FilePatch]:
    patches: list[FilePatch] = []
    for proposal in proposals:
        if hasattr(proposal, "file_patches"):
            patches.extend(proposal.file_patches)
    return patches
