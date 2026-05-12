"""HART/MobileHART-style lightweight transformer for IMU HAR."""

from __future__ import annotations

try:
    import torch
except Exception:  # pragma: no cover
    torch = None

from research.modules.architectures.common import as_sequence


ARCHITECTURE_NAME = "hart"


class HARTBlock(torch.nn.Module if torch is not None else object):
    def __init__(self, hidden_dim: int, heads: int, dropout: float):
        super().__init__()
        self.norm1 = torch.nn.LayerNorm(hidden_dim)
        self.attn = torch.nn.MultiheadAttention(
            hidden_dim, heads, dropout=dropout, batch_first=True
        )
        self.norm2 = torch.nn.LayerNorm(hidden_dim)
        self.ffn = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, hidden_dim * 2),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_dim * 2, hidden_dim),
        )

    def forward(self, x):
        attn_out, _ = self.attn(self.norm1(x), self.norm1(x), self.norm1(x))
        x = x + attn_out
        return x + self.ffn(self.norm2(x))


class HART(torch.nn.Module if torch is not None else object):
    def __init__(self, input_dim: int, num_classes: int, params: dict[str, object]):
        super().__init__()
        self.input_channels = int(params.get("input_channels", 1))
        hidden_dim = int(params.get("hidden_dim", 72))
        depth = max(1, int(params.get("depth", 2)))
        heads = max(1, int(params.get("heads", 4)))
        if hidden_dim % heads != 0:
            heads = 1
        dropout = float(params.get("dropout", 0.0))
        self.temporal_patch = torch.nn.Conv1d(
            self.input_channels, hidden_dim, kernel_size=5, stride=2, padding=2
        )
        self.sensor_gate = torch.nn.Sequential(
            torch.nn.AdaptiveAvgPool1d(1),
            torch.nn.Flatten(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.Sigmoid(),
        )
        self.blocks = torch.nn.Sequential(
            *[HARTBlock(hidden_dim, heads, dropout) for _ in range(depth)]
        )
        self.classifier = torch.nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        sequence = as_sequence(x, self.input_channels).transpose(1, 2)
        tokens = self.temporal_patch(sequence)
        gate = self.sensor_gate(tokens).unsqueeze(-1)
        tokens = (tokens * gate).transpose(1, 2)
        encoded = self.blocks(tokens)
        return self.classifier(encoded.mean(dim=1))


def build_model(input_dim: int, num_classes: int, params: dict[str, object]):
    del input_dim
    if torch is None:
        raise RuntimeError("PyTorch is required for hart")
    return HART(input_dim=1, num_classes=num_classes, params=params)
