"""Smooth magnitude warping."""

from __future__ import annotations

from research.modules.augmentations.common import strength_scale

NAME = "magnitude_warp"


def augment_window(window, strength, params, rng):
    scale = strength_scale(strength)
    anchors = [0.0, 0.33, 0.66, 1.0]
    span = float(params.get("warp_span", 0.15 * scale))
    knots = [rng.uniform(1.0 - span, 1.0 + span) for _ in anchors]
    warped = []
    for step, row in enumerate(window):
        alpha = step / max(len(window) - 1, 1)
        left = max(index for index, anchor in enumerate(anchors) if anchor <= alpha)
        right = min(left + 1, len(anchors) - 1)
        if left == right:
            factor = knots[left]
        else:
            width = anchors[right] - anchors[left]
            mix = 0.0 if width == 0 else (alpha - anchors[left]) / width
            factor = ((1.0 - mix) * knots[left]) + (mix * knots[right])
        warped.append([value * factor for value in row])
    return warped
