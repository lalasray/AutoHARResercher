"""2D horizontal-plane rotation for accelerometer and gyro channels."""

from __future__ import annotations

import math

from research.modules.augmentations.common import strength_scale

NAME = "rotation"


def augment_window(window, strength, params, rng):
    del rng
    scale = strength_scale(strength)
    angle = math.radians(float(params.get("angle_deg", 10.0 * scale)))
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)
    rotated = []
    for row in window:
        ax, ay, az, gx, gy, gz = row[:6]
        rotated.append(
            [
                cos_a * ax - sin_a * ay,
                sin_a * ax + cos_a * ay,
                az,
                cos_a * gx - sin_a * gy,
                sin_a * gx + cos_a * gy,
                gz,
                *row[6:],
            ]
        )
    return rotated
