"""Configurable custom feature head for bounded custom representations."""

from __future__ import annotations

try:
    import torch
except Exception:  # pragma: no cover
    torch = None


ARCHITECTURE_NAME = "custom"


class CustomFeatureHead(torch.nn.Module if torch is not None else object):
    def __init__(self, input_dim: int, num_classes: int, params: dict[str, object]):
        super().__init__()
        hidden_dim = int(params.get("hidden_dim", 64))
        depth = max(1, int(params.get("depth", 2)))
        dropout = float(params.get("dropout", 0.0))
        layers: list[torch.nn.Module] = []
        width = input_dim
        for _ in range(depth):
            layers.extend(
                [
                    torch.nn.Linear(width, hidden_dim),
                    torch.nn.ReLU(),
                    torch.nn.Dropout(dropout),
                ]
            )
            width = hidden_dim
        layers.append(torch.nn.Linear(width, num_classes))
        self.net = torch.nn.Sequential(*layers)

    def forward(self, x):
        if x.dim() > 2:
            x = x.mean(dim=1)
        return self.net(x)


def build_model(input_dim: int, num_classes: int, params: dict[str, object]):
    if torch is None:
        raise RuntimeError("PyTorch is required for custom")
    return CustomFeatureHead(input_dim=input_dim, num_classes=num_classes, params=params)
