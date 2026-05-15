"""Bounded IMU-to-IMU proxy augmentation.

This module is intentionally conservative: it creates small same-shape sensor
variations that are useful as a stand-in for future learned generators.
"""

from __future__ import annotations

import math

from research.modules.augmentations.common import strength_scale

NAME = "imu_to_imu"

DEFAULT_VARIATION_AMOUNT = 0.12
MAX_VARIATION_AMOUNT = 0.5
NOISE_FRACTION = 0.05
BIAS_FRACTION = 0.03
SLOW_WAVE_FRACTION = 0.02


def _variation_amount(strength: str, params: dict[str, object]) -> float:
    scaled_default = DEFAULT_VARIATION_AMOUNT * strength_scale(strength)
    requested = float(params.get("variation_amount", scaled_default))
    return max(0.0, min(MAX_VARIATION_AMOUNT, requested))


def _channel_biases(channel_count: int, amount: float, rng) -> list[float]:
    bias_radius = amount * BIAS_FRACTION
    return [rng.uniform(-bias_radius, bias_radius) for _ in range(channel_count)]


def _slow_wave(step: int, window_length: int, amount: float) -> float:
    alpha = step / max(window_length - 1, 1)
    return math.sin(alpha * math.pi * 2.0) * amount * SLOW_WAVE_FRACTION


def augment_window(window, strength, params, rng):
    if not window:
        return []
    amount = _variation_amount(strength, params)
    noise_radius = amount * NOISE_FRACTION
    biases = _channel_biases(len(window[0]), amount, rng)
    out = []
    for step, row in enumerate(window):
        wave = _slow_wave(step, len(window), amount)
        out.append(
            [
                value
                + biases[channel]
                + wave
                + rng.uniform(-noise_radius, noise_radius)
                for channel, value in enumerate(row)
            ]
        )
    return out
