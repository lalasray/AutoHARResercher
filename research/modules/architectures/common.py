"""Shared building blocks for HAR architecture modules."""

from __future__ import annotations

def as_sequence(x, input_channels: int = 1):
    """Return x as [batch, time, channels]."""
    if x.dim() != 3 or x.shape[-1] != input_channels:
        raise ValueError("Expected [batch, time, configured channels]")
    return x
