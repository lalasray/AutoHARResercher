"""Window-level normalization preprocessing."""

from __future__ import annotations

from research.modules.preprocessing.common import normalize

NAME = "normalized"


def prepare_window(window, params):
    if bool(params.get("normalize", True)):
        return normalize(window)
    return [row[:] for row in window]
