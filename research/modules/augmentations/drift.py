"""Linear sensor drift."""

from __future__ import annotations

from research.modules.augmentations.common import strength_scale

NAME = "drift"


def augment_window(window, strength, params, rng):
    scale = strength_scale(strength)
    radius = float(params.get("drift_max", 0.015 * scale))
    drifts = [rng.uniform(-radius, radius) for _ in range(len(window[0]))]
    out = []
    for step, row in enumerate(window):
        alpha = step / max(len(window) - 1, 1)
        out.append([value + alpha * drifts[idx] for idx, value in enumerate(row)])
    return out
