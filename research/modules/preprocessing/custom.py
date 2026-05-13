"""Custom preprocessing placeholder module."""

from __future__ import annotations

from research.modules.preprocessing.common import copy_window

NAME = "custom"


def prepare_window(window, params):
    del params
    return copy_window(window)
