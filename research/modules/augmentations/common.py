"""Shared helpers for augmentation modules."""

from __future__ import annotations

import math


def copy_window(window: list[list[float]]) -> list[list[float]]:
    return [row[:] for row in window]


def strength_scale(strength: str) -> float:
    return {"weak": 0.5, "medium": 1.0, "strong": 1.5}.get(strength, 1.0)


def validate_window(candidate: list[list[float]], fallback: list[list[float]]) -> list[list[float]]:
    if len(candidate) != len(fallback):
        return copy_window(fallback)
    channels = len(fallback[0]) if fallback else 0
    clean: list[list[float]] = []
    for row in candidate:
        if len(row) != channels:
            return copy_window(fallback)
        clean_row = []
        for value in row:
            if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                return copy_window(fallback)
            clean_row.append(float(value))
        clean.append(clean_row)
    return clean
