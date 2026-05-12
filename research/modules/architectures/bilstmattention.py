"""Attention-based BiLSTM for classification."""

from __future__ import annotations

try:
    import torch
except Exception:  # pragma: no cover
    torch = None

from research.modules.architectures.common import TemporalAttention, as_sequence


ARCHITECTURE_NAME = "bilstmattention"


class BiLSTMAttention(torch.nn.Module if torch is not None else object):
    def __init__(self, input_dim: int, num_classes: int, params: dict[str, object]):
        super().__init__()
        self.input_channels = int(params.get("input_channels", 1))
        hidden_dim = int(params.get("hidden_dim", 72))
        layers = max(1, int(params.get("depth", 2)))
        dropout = float(params.get("dropout", 0.1))
        bidirectional = bool(params.get("bidirectional", True))
        self.lstm = torch.nn.LSTM(
            input_size=self.input_channels,
            hidden_size=hidden_dim,
            num_layers=layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if layers > 1 else 0.0,
        )
        width = hidden_dim * (2 if bidirectional else 1)
        self.attention = TemporalAttention(width)
        self.classifier = torch.nn.Linear(width, num_classes)

    def forward(self, x):
        sequence = as_sequence(x, self.input_channels)
        recurrent, _ = self.lstm(sequence)
        return self.classifier(self.attention(recurrent))


def build_model(input_dim: int, num_classes: int, params: dict[str, object]):
    del input_dim
    if torch is None:
        raise RuntimeError("PyTorch is required for bilstmattention")
    return BiLSTMAttention(input_dim=1, num_classes=num_classes, params=params)
