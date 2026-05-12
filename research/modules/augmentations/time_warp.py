"""Simple temporal compression warp."""

from __future__ import annotations

from research.modules.augmentations.common import strength_scale

NAME = "time_warp"


def augment_window(window, strength, params, rng):
    del rng
    scale = strength_scale(strength)
    compression = float(params.get("compression_ratio", max(0.7, 1.0 - (0.08 * scale))))
    return [window[min(len(window) - 1, int(index * compression))][:] for index in range(len(window))]
