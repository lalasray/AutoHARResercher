"""Composed placeholder IMU-to-IMU augmentation."""

from __future__ import annotations

import math

from research.modules.augmentations.common import strength_scale

NAME = "imu_to_imu"


def augment_window(window, strength, params, rng):
    scale = strength_scale(strength)
    amount = max(0.0, min(0.5, float(params.get("variation_amount", 0.12 * scale))))
    noise_radius = amount * 0.05
    bias_radius = amount * 0.03
    biases = [rng.uniform(-bias_radius, bias_radius) for _ in range(len(window[0]))]
    out = []
    for step, row in enumerate(window):
        alpha = step / max(len(window) - 1, 1)
        slow_wave = math.sin(alpha * math.pi * 2.0) * amount * 0.02
        out.append(
            [
                value + biases[channel] + slow_wave + rng.uniform(-noise_radius, noise_radius)
                for channel, value in enumerate(row)
            ]
        )
    return out
