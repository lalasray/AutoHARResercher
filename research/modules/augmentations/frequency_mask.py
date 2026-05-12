"""Temporal masking proxy inspired by frequency/time masking."""

from __future__ import annotations

from research.modules.augmentations.common import copy_window, strength_scale

NAME = "frequency_mask"


def augment_window(window, strength, params, rng):
    scale = strength_scale(strength)
    ratio = max(0.0, min(0.6, float(params.get("mask_ratio", 0.12 * scale))))
    if len(window) < 4:
        return copy_window(window)
    span = max(1, int(len(window) * ratio))
    start = int(rng.uniform(0, max(1, len(window) - span)))
    masked = copy_window(window)
    means = [sum(row[channel] for row in window) / len(window) for channel in range(len(window[0]))]
    for idx in range(start, min(len(masked), start + span)):
        masked[idx] = means[:]
    return masked
