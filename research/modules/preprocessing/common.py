"""Shared preprocessing helpers."""

from __future__ import annotations

import math


def copy_window(window: list[list[float]]) -> list[list[float]]:
    return [row[:] for row in window]


def normalize(window: list[list[float]]) -> list[list[float]]:
    channels = len(window[0])
    means = [sum(row[i] for row in window) / len(window) for i in range(channels)]
    stds = []
    for i in range(channels):
        variance = sum((row[i] - means[i]) ** 2 for row in window) / len(window)
        stds.append(math.sqrt(variance) + 1e-6)
    return [[(row[i] - means[i]) / stds[i] for i in range(channels)] for row in window]


def append_magnitude(window: list[list[float]]) -> list[list[float]]:
    extended: list[list[float]] = []
    for row in window:
        accel_mag = math.sqrt(sum(value * value for value in row[:3]))
        gyro_mag = math.sqrt(sum(value * value for value in row[3:6]))
        extended.append(row + [accel_mag, gyro_mag])
    return extended
