"""Deep temporal CNN baseline for wearable HAR."""

from __future__ import annotations

try:
    import torch
except Exception:  # pragma: no cover
    torch = None

from research.modules.architectures.common import as_sequence


ARCHITECTURE_NAME = "cnn"


class CNN(torch.nn.Module if torch is not None else object):
    def __init__(self, input_dim: int, num_classes: int, params: dict[str, object]):
        super().__init__()
        self.input_channels = int(params.get("input_channels", 1))
        filters = int(params.get("hidden_dim", 32))
        depth = max(1, int(params.get("depth", 4)))
        layers: list[torch.nn.Module] = []
        channels = self.input_channels
        for _ in range(depth):
            layers.extend(
                [
                    torch.nn.Conv1d(channels, filters, kernel_size=5, padding=2),
                    torch.nn.ReLU(),
                ]
            )
            channels = filters
        self.conv = torch.nn.Sequential(*layers)
        self.classifier = torch.nn.Linear(filters, num_classes)

    def forward(self, x):
        sequence = as_sequence(x, self.input_channels).transpose(1, 2)
        features = self.conv(sequence).mean(dim=-1)
        return self.classifier(features)


def build_model(input_dim: int, num_classes: int, params: dict[str, object]):
    del input_dim
    if torch is None:
        raise RuntimeError("PyTorch is required for cnn")
    return CNN(input_dim=1, num_classes=num_classes, params=params)
