"""Per-channel magnitude scaling."""

from __future__ import annotations

from research.modules.augmentations.common import strength_scale

NAME = "scaling"


def augment_window(window, strength, params, rng):
    scale = strength_scale(strength)
    span = float(params.get("scale_span", 0.1 * scale))
    factors = [rng.uniform(1.0 - span, 1.0 + span) for _ in range(len(window[0]))]
    return [[value * factors[idx] for idx, value in enumerate(row)] for row in window]
