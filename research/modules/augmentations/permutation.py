"""Permute contiguous temporal segments."""

from __future__ import annotations

from research.modules.augmentations.common import copy_window, strength_scale

NAME = "permutation"


def augment_window(window, strength, params, rng):
    if len(window) < 8:
        return copy_window(window)
    scale = strength_scale(strength)
    segment_count = int(params.get("segment_count", min(6, max(3, int(round(2 + (2 * scale)))))))
    segment_size = max(len(window) // segment_count, 1)
    segments = [window[index : index + segment_size] for index in range(0, len(window), segment_size)]
    rng.shuffle(segments)
    return [row[:] for segment in segments for row in segment]
