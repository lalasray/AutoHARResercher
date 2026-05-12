"""Custom augmentation placeholder module.

Runtime custom code is still loaded by the harness sandbox; this module exists so
the augmentation family has the same file-based structure as the others.
"""

from __future__ import annotations

from research.modules.augmentations.common import copy_window

NAME = "custom"


def augment_window(window, strength, params, rng):
    del strength, params, rng
    return copy_window(window)
