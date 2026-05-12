"""Add fixed per-channel sensor bias."""

from __future__ import annotations

from research.modules.augmentations.common import strength_scale

NAME = "sensor_bias"


def augment_window(window, strength, params, rng):
    scale = strength_scale(strength)
    radius = float(params.get("bias_radius", 0.02 * scale))
    biases = [rng.uniform(-radius, radius) for _ in range(len(window[0]))]
    return [[value + biases[idx] for idx, value in enumerate(row)] for row in window]
