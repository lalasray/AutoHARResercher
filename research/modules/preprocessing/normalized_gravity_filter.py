"""Normalization plus simple gravity mean removal."""

from __future__ import annotations

from research.modules.preprocessing.common import normalize

NAME = "normalized+gravity_filter"


def prepare_window(window, params):
    processed = normalize(window) if bool(params.get("normalize", True)) else [row[:] for row in window]
    gravity_scale = float(params.get("gravity_scale", 1.0))
    means = [sum(row[channel] for row in processed) / len(processed) for channel in range(3)]
    for row in processed:
        for channel in range(3):
            row[channel] -= gravity_scale * means[channel]
    return processed
