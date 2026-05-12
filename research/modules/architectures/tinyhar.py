"""TinyHAR-style lightweight model from ISWC22-HAR.

Implements the key TinyHAR stages: per-channel temporal convolution with stride
2, cross-channel transformer interaction, FC channel fusion, one-layer LSTM,
and temporal attention with trainable gamma.
"""

from __future__ import annotations

try:
    import torch
except Exception:  # pragma: no cover
    torch = None

from research.modules.architectures.common import TemporalAttention, as_sequence


ARCHITECTURE_NAME = "tinyhar"


class TinyHAR(torch.nn.Module if torch is not None else object):
    def __init__(self, input_dim: int, num_classes: int, params: dict[str, object]):
        super().__init__()
        self.input_channels = int(params.get("input_channels", 1))
        filters = int(params.get("hidden_dim", 40))
        dropout = float(params.get("dropout", 0.0))
        conv_layers: list[torch.nn.Module] = []
        in_channels = self.input_channels
        out_channels = self.input_channels * filters
        for idx in range(4):
            conv_layers.extend(
                [
                    torch.nn.Conv1d(
                        in_channels,
                        out_channels,
                        kernel_size=5,
                        stride=2,
                        padding=2,
                        groups=self.input_channels,
                    ),
                    torch.nn.BatchNorm1d(out_channels),
                    torch.nn.ReLU(),
                ]
            )
            in_channels = out_channels
        self.channel_conv = torch.nn.Sequential(*conv_layers)
        heads = 1 if filters < 4 else min(4, filters)
        if filters % heads != 0:
            heads = 1
        layer = torch.nn.TransformerEncoderLayer(
            d_model=filters,
            nhead=heads,
            dim_feedforward=filters * 2,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.channel_interaction = torch.nn.TransformerEncoder(layer, num_layers=1)
        self.fusion = torch.nn.Linear(self.input_channels * filters, filters * 2)
        self.lstm = torch.nn.LSTM(
            input_size=filters * 2,
            hidden_size=filters * 2,
            num_layers=1,
            batch_first=True,
        )
        self.attention = TemporalAttention(filters * 2)
        self.gamma = torch.nn.Parameter(torch.tensor(0.0))
        self.classifier = torch.nn.Linear(filters * 2, num_classes)

    def forward(self, x):
        sequence = as_sequence(x, self.input_channels).transpose(1, 2)
        conv = self.channel_conv(sequence)
        batch, _, time_steps = conv.shape
        conv = conv.reshape(batch, self.input_channels, -1, time_steps)
        conv = conv.permute(0, 3, 1, 2)
        interacted = self.channel_interaction(conv.reshape(batch * time_steps, self.input_channels, -1))
        interacted = interacted.reshape(batch, time_steps, self.input_channels, -1)
        fused = self.fusion(interacted.flatten(start_dim=2))
        recurrent, _ = self.lstm(fused)
        context = self.attention(recurrent)
        final = recurrent[:, -1, :] + self.gamma * context
        return self.classifier(final)


def build_model(input_dim: int, num_classes: int, params: dict[str, object]):
    del input_dim
    if torch is None:
        raise RuntimeError("PyTorch is required for tinyhar")
    return TinyHAR(input_dim=1, num_classes=num_classes, params=params)
