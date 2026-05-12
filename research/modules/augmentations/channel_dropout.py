"""Drop whole sensor channels."""

from __future__ import annotations

from research.modules.augmentations.common import strength_scale

NAME = "channel_dropout"


def augment_window(window, strength, params, rng):
    scale = strength_scale(strength)
    probability = float(params.get("drop_probability", 0.08 * scale))
    channel_count = len(window[0]) if window else 0
    dropped = {
        channel for channel in range(channel_count) if rng.uniform(0.0, 1.0) < probability
    }
    if not dropped and channel_count:
        dropped.add(int(rng.uniform(0, channel_count)))
    return [
        [0.0 if channel in dropped else value for channel, value in enumerate(row)]
        for row in window
    ]
