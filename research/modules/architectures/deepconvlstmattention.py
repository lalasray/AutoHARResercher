"""DeepConvLSTM plus context-vector temporal attention."""

from __future__ import annotations

try:
    import torch
except Exception:  # pragma: no cover
    torch = None

from research.modules.architectures.common import TemporalAttention, as_sequence


ARCHITECTURE_NAME = "deepconvlstmattention"


class DeepConvLSTMAttention(torch.nn.Module if torch is not None else object):
    def __init__(self, input_dim: int, num_classes: int, params: dict[str, object]):
        super().__init__()
        self.input_channels = int(params.get("input_channels", 1))
        conv_filters = int(params.get("conv_filters", params.get("hidden_dim", 80)))
        lstm_units = int(params.get("hidden_dim", 80))
        lstm_layers = max(1, int(params.get("lstm_layers", 2)))
        dropout = float(params.get("dropout", 0.0))
        conv_layers: list[torch.nn.Module] = []
        channels = self.input_channels
        for _ in range(4):
            conv_layers.extend(
                [
                    torch.nn.Conv1d(channels, conv_filters, kernel_size=5, padding=2),
                    torch.nn.ReLU(),
                ]
            )
            channels = conv_filters
        self.conv = torch.nn.Sequential(*conv_layers)
        self.lstm = torch.nn.LSTM(
            input_size=conv_filters,
            hidden_size=lstm_units,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )
        self.attention = TemporalAttention(lstm_units)
        self.classifier = torch.nn.Linear(lstm_units, num_classes)

    def forward(self, x):
        sequence = as_sequence(x, self.input_channels).transpose(1, 2)
        features = self.conv(sequence).transpose(1, 2)
        recurrent, _ = self.lstm(features)
        return self.classifier(self.attention(recurrent))


def build_model(input_dim: int, num_classes: int, params: dict[str, object]):
    del input_dim
    if torch is None:
        raise RuntimeError("PyTorch is required for deepconvlstmattention")
    return DeepConvLSTMAttention(input_dim=1, num_classes=num_classes, params=params)
