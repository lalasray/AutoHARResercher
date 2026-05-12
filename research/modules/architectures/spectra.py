"""Spectral-temporal fusion model for wearable HAR."""

from __future__ import annotations

try:
    import torch
except Exception:  # pragma: no cover
    torch = None

from research.modules.architectures.common import as_sequence


ARCHITECTURE_NAME = "spectra"


class SpectraNet(torch.nn.Module if torch is not None else object):
    def __init__(self, input_dim: int, num_classes: int, params: dict[str, object]):
        super().__init__()
        self.input_channels = int(params.get("input_channels", 1))
        hidden_dim = int(params.get("hidden_dim", 56))
        self.spectral_bins = max(2, int(params.get("spectral_bins", 10)))
        dropout = float(params.get("dropout", 0.0))
        self.temporal_conv = torch.nn.Sequential(
            torch.nn.Conv1d(self.input_channels, hidden_dim, kernel_size=5, padding=2),
            torch.nn.ReLU(),
            torch.nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            torch.nn.ReLU(),
        )
        self.spectral_proj = torch.nn.Linear(self.input_channels * self.spectral_bins, hidden_dim)
        self.fusion = torch.nn.Sequential(
            torch.nn.LayerNorm(hidden_dim * 2),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_dim * 2, hidden_dim),
            torch.nn.SiLU(),
            torch.nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x):
        sequence = as_sequence(x, self.input_channels)
        channels_first = sequence.transpose(1, 2)
        temporal = self.temporal_conv(channels_first).mean(dim=-1)
        spectrum = torch.fft.rfft(sequence, dim=1).abs().transpose(1, 2)
        if spectrum.shape[-1] < self.spectral_bins:
            spectrum = torch.nn.functional.pad(
                spectrum, (0, self.spectral_bins - spectrum.shape[-1])
            )
        spectrum = spectrum[:, :, : self.spectral_bins].flatten(start_dim=1)
        spectral = torch.relu(self.spectral_proj(spectrum))
        return self.fusion(torch.cat([temporal, spectral], dim=-1))


def build_model(input_dim: int, num_classes: int, params: dict[str, object]):
    del input_dim
    if torch is None:
        raise RuntimeError("PyTorch is required for spectra")
    return SpectraNet(input_dim=1, num_classes=num_classes, params=params)
