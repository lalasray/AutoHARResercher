"""Add bounded sensor jitter."""

from __future__ import annotations

from research.modules.augmentations.common import strength_scale

NAME = "jitter"


def augment_window(window, strength, params, rng):
    scale = strength_scale(strength)
    radius = float(params.get("jitter_std", 0.02 * scale))
    return [[value + rng.uniform(-radius, radius) for value in row] for row in window]
