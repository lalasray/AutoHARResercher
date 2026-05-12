"""No augmentation."""

from __future__ import annotations

from research.modules.augmentations.common import copy_window

NAME = "none"


def augment_window(window, strength, params, rng):
    del strength, params, rng
    return copy_window(window)
