"""Normalization plus accelerometer/gyro magnitude channels."""

from __future__ import annotations

from research.modules.preprocessing.common import append_magnitude, normalize

NAME = "normalized+magnitude_channels"


def prepare_window(window, params):
    processed = normalize(window) if bool(params.get("normalize", True)) else [row[:] for row in window]
    if bool(params.get("append_magnitude", True)):
        processed = append_magnitude(processed)
    return processed
