"""TinierHAR-style ultra-lightweight HAR architecture.

Uses residual depthwise separable temporal convolutions, a compact GRU, and
attention-style temporal aggregation.
"""

from __future__ import annotations

try:
    import torch
except Exception:  # pragma: no cover
    torch = None

from research.modules.architectures.common import TemporalAttention, as_sequence


ARCHITECTURE_NAME = "tinierhar"


class DepthwiseSeparableBlock(torch.nn.Module if torch is not None else object):
    def __init__(self, channels: int, dropout: float):
        super().__init__()
        self.block = torch.nn.Sequential(
            torch.nn.Conv1d(channels, channels, kernel_size=5, padding=2, groups=channels),
            torch.nn.BatchNorm1d(channels),
            torch.nn.ReLU(),
            torch.nn.Conv1d(channels, channels, kernel_size=1),
            torch.nn.Dropout(dropout),
        )

    def forward(self, x):
        return torch.relu(x + self.block(x))


class TinierHAR(torch.nn.Module if torch is not None else object):
    def __init__(self, input_dim: int, num_classes: int, params: dict[str, object]):
        super().__init__()
        self.input_channels = int(params.get("input_channels", 1))
        hidden_dim = int(params.get("hidden_dim", 32))
        depth = max(1, int(params.get("depth", 1)))
        dropout = float(params.get("dropout", 0.0))
        self.stem = torch.nn.Conv1d(self.input_channels, hidden_dim, kernel_size=3, padding=1)
        self.blocks = torch.nn.Sequential(
            *[DepthwiseSeparableBlock(hidden_dim, dropout) for _ in range(depth)]
        )
        self.gru = torch.nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
        )
        self.attention = TemporalAttention(hidden_dim)
        self.classifier = torch.nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        sequence = as_sequence(x, self.input_channels).transpose(1, 2)
        temporal = self.blocks(torch.relu(self.stem(sequence))).transpose(1, 2)
        recurrent, _ = self.gru(temporal)
        return self.classifier(self.attention(recurrent))


def build_model(input_dim: int, num_classes: int, params: dict[str, object]):
    del input_dim
    if torch is None:
        raise RuntimeError("PyTorch is required for tinierhar")
    return TinierHAR(input_dim=1, num_classes=num_classes, params=params)
