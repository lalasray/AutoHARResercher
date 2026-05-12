"""Window self-mixup proxy."""

from __future__ import annotations

from research.modules.augmentations.common import strength_scale

NAME = "mixup"


def augment_window(window, strength, params, rng):
    del rng
    scale = strength_scale(strength)
    blend = float(params.get("blend", min(0.45, 0.15 + (0.15 * scale))))
    return [
        [((1.0 - blend) * a) + (blend * b) for a, b in zip(left, right)]
        for left, right in zip(window, reversed(window))
    ]
