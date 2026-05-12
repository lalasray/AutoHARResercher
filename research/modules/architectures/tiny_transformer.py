"""Small transformer encoder baseline for wearable-sensor HAR."""

from __future__ import annotations

import math

try:
    import torch
except Exception:  # pragma: no cover
    torch = None

from research.modules.architectures.common import as_sequence


ARCHITECTURE_NAME = "tiny_transformer"


class TinyTransformer(torch.nn.Module if torch is not None else object):
    def __init__(self, input_dim: int, num_classes: int, params: dict[str, object]):
        super().__init__()
        self.input_channels = int(params.get("input_channels", 1))
        hidden_dim = int(params.get("hidden_dim", 48))
        depth = max(1, int(params.get("depth", 2)))
        heads = max(1, int(params.get("heads", 4)))
        if hidden_dim % heads != 0:
            heads = 1
        self.embedding = torch.nn.Linear(self.input_channels, hidden_dim)
        layer = torch.nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=heads,
            dim_feedforward=hidden_dim * 2,
            dropout=float(params.get("dropout", 0.0)),
            batch_first=True,
            activation="gelu",
        )
        self.encoder = torch.nn.TransformerEncoder(layer, num_layers=depth)
        self.classifier = torch.nn.Linear(hidden_dim, num_classes)

    def _positional_encoding(self, length: int, dim: int, device):
        positions = torch.arange(length, device=device).float().unsqueeze(1)
        div = torch.exp(
            torch.arange(0, dim, 2, device=device).float() * (-math.log(10000.0) / dim)
        )
        pe = torch.zeros(length, dim, device=device)
        pe[:, 0::2] = torch.sin(positions * div)
        pe[:, 1::2] = torch.cos(positions * div[: pe[:, 1::2].shape[1]])
        return pe.unsqueeze(0)

    def forward(self, x):
        sequence = as_sequence(x, self.input_channels)
        tokens = self.embedding(sequence)
        tokens = tokens + self._positional_encoding(
            tokens.shape[1], tokens.shape[2], tokens.device
        )
        encoded = self.encoder(tokens)
        return self.classifier(encoded.mean(dim=1))


def build_model(input_dim: int, num_classes: int, params: dict[str, object]):
    del input_dim
    if torch is None:
        raise RuntimeError("PyTorch is required for tiny_transformer")
    return TinyTransformer(input_dim=1, num_classes=num_classes, params=params)
