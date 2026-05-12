"""Convolutional GRU hybrid for wearable HAR."""

from __future__ import annotations

try:
    import torch
except Exception:  # pragma: no cover
    torch = None

from research.modules.architectures.common import as_sequence


ARCHITECTURE_NAME = "cnn_gru"


class CNNGRU(torch.nn.Module if torch is not None else object):
    def __init__(self, input_dim: int, num_classes: int, params: dict[str, object]):
        super().__init__()
        self.input_channels = int(params.get("input_channels", 1))
        hidden_dim = int(params.get("hidden_dim", 56))
        depth = max(1, int(params.get("depth", 2)))
        dropout = float(params.get("dropout", 0.0))
        self.conv = torch.nn.Sequential(
            torch.nn.Conv1d(self.input_channels, hidden_dim, kernel_size=5, padding=2),
            torch.nn.ReLU(),
            torch.nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            torch.nn.ReLU(),
        )
        self.gru = torch.nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=depth,
            batch_first=True,
            dropout=dropout if depth > 1 else 0.0,
        )
        self.classifier = torch.nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        sequence = as_sequence(x, self.input_channels).transpose(1, 2)
        tokens = self.conv(sequence).transpose(1, 2)
        _, hidden = self.gru(tokens)
        return self.classifier(hidden[-1])


def build_model(input_dim: int, num_classes: int, params: dict[str, object]):
    del input_dim
    if torch is None:
        raise RuntimeError("PyTorch is required for cnn_gru")
    return CNNGRU(input_dim=1, num_classes=num_classes, params=params)
