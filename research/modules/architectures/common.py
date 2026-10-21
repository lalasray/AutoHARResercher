"""Shared building blocks for HAR architecture modules."""

from __future__ import annotations

try:
    import torch
except Exception:  # pragma: no cover
    torch = None


def as_sequence(x, input_channels: int = 1):
    """Return x as [batch, time, channels]."""
    if x.dim() != 3 or x.shape[-1] != input_channels:
        raise ValueError("Expected [batch, time, configured channels]")
    return x



class TemporalAttention(torch.nn.Module if torch is not None else object):
    """Context-vector temporal attention for sequence classification."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.proj = torch.nn.Linear(hidden_dim, hidden_dim)
        self.context = torch.nn.Parameter(torch.empty(hidden_dim))
        torch.nn.init.xavier_uniform_(self.context.unsqueeze(0))

    def forward(self, sequence):
        scores = torch.tanh(self.proj(sequence)) @ self.context
        weights = torch.softmax(scores, dim=1).unsqueeze(-1)
        return (sequence * weights).sum(dim=1)
