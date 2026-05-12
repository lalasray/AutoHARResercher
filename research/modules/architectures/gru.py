"""GRU sequence classifier."""

from __future__ import annotations

try:
    import torch
except Exception:  # pragma: no cover
    torch = None

from research.modules.architectures.common import as_sequence


ARCHITECTURE_NAME = "gru"


class GRU(torch.nn.Module if torch is not None else object):
    def __init__(self, input_dim: int, num_classes: int, params: dict[str, object]):
        super().__init__()
        self.input_channels = int(params.get("input_channels", 1))
        hidden_dim = int(params.get("hidden_dim", 40))
        layers = max(1, int(params.get("depth", 1)))
        dropout = float(params.get("dropout", 0.0))
        self.gru = torch.nn.GRU(
            input_size=self.input_channels,
            hidden_size=hidden_dim,
            num_layers=layers,
            batch_first=True,
            dropout=dropout if layers > 1 else 0.0,
        )
        self.classifier = torch.nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        sequence = as_sequence(x, self.input_channels)
        _, hidden = self.gru(sequence)
        return self.classifier(hidden[-1])


def build_model(input_dim: int, num_classes: int, params: dict[str, object]):
    del input_dim
    if torch is None:
        raise RuntimeError("PyTorch is required for gru")
    return GRU(input_dim=1, num_classes=num_classes, params=params)
