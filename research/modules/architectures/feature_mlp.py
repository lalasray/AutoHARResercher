"""Local experimental statistics/FFT MLP; not a published author baseline.

Accepts ordinary sensor windows. Feature scaling is fitted explicitly on the
training partition and stored in checkpoint buffers; forward never fits it.
This Torch frontend is a reusable implementation of the experimental family,
not a drop-in loader for the earlier NumPy-feature reproduction checkpoints.
"""
import math
import torch
from .common import as_sequence

ARCHITECTURE_NAME = 'feature_mlp'


class FeatureMLP(torch.nn.Module):
    def __init__(self, input_dim, num_classes, params):
        super().__init__()
        self.input_channels = int(params.get('input_channels', 1))
        self.sampling_hz = float(params.get('sampling_hz', 50))
        if not math.isfinite(self.sampling_hz) or self.sampling_hz <= 0:
            raise ValueError('sampling_hz must be finite and positive')
        self.steps = input_dim // self.input_channels
        if self.steps < 2 or self.steps * self.input_channels != input_dim:
            raise ValueError('Expected at least two samples per configured channel')
        names = tuple(params.get('channel_names', ()))
        self.magnitude_groups = []
        for group in [('ax', 'ay', 'az'), ('gx', 'gy', 'gz')]:
            if all(name in names for name in group):
                self.magnitude_groups.append(tuple(names.index(name) for name in group))
        self.frequency_bands = [(lo, min(hi, self.sampling_hz / 2))
                                for lo, hi in zip([0, .5, 1, 2, 3, 5, 8, 12, 20],
                                                  [.5, 1, 2, 3, 5, 8, 12, 20, math.inf])
                                if lo < self.sampling_hz / 2]
        size = self.extract_features(torch.zeros(1, self.steps, self.input_channels)).shape[1]
        self.register_buffer('feature_mean', torch.zeros(size))
        self.register_buffer('feature_scale', torch.ones(size))
        self.register_buffer('statistics_fitted', torch.tensor(False))
        width = int(params.get('hidden_dim', 256))
        dropout = float(params.get('dropout', .3))
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(size, width), torch.nn.LayerNorm(width), torch.nn.SiLU(),
            torch.nn.Dropout(dropout), torch.nn.Linear(width, width // 2),
            torch.nn.LayerNorm(width // 2), torch.nn.SiLU(), torch.nn.Dropout(dropout),
            torch.nn.Linear(width // 2, num_classes))

    def extract_features(self, x):
        x = as_sequence(x, self.input_channels)
        if x.shape[1] != self.steps:
            raise ValueError('Window length differs from configured input_dim')
        if self.magnitude_groups:
            x = torch.cat([x] + [x[:, :, group].norm(dim=2, keepdim=True)
                                 for group in self.magnitude_groups], dim=2)
        mean = x.mean(1)
        centered = x - mean[:, None, :]
        variance = centered.square().mean(1)
        std = variance.sqrt()
        q = torch.quantile(x, torch.tensor([.25, .5, .75], device=x.device), dim=1)
        delta = x[:, 1:] - x[:, :-1]
        parts = [mean, std, x.square().mean(1).sqrt(), x.amin(1), x.amax(1),
                 q[0], q[1], q[2], q[2] - q[0], x.abs().mean(1),
                 centered.abs().mean(1), centered.pow(3).mean(1) / std.clamp_min(1e-6).pow(3),
                 centered.pow(4).mean(1) / variance.clamp_min(1e-6).square(),
                 delta.std(1, unbiased=False), delta.abs().mean(1),
                 ((centered[:, 1:] >= 0) != (centered[:, :-1] >= 0)).float().mean(1)]
        for seconds in [.05, .25, .5, 1.]:
            lag = max(1, round(seconds * self.sampling_hz))
            if lag < self.steps:
                parts.append((centered[:, :-lag] * centered[:, lag:]).mean(1)
                             / variance.clamp_min(1e-6))
        power = torch.fft.rfft(centered, dim=1).abs().square()
        frequencies = torch.fft.rfftfreq(self.steps, d=1 / self.sampling_hz, device=x.device)
        total = power.sum(1).clamp_min(1e-6)
        probability = power / total[:, None, :]
        parts += [(power * frequencies[None, :, None]).sum(1) / total,
                  -(probability * probability.clamp_min(1e-12).log()).sum(1)]
        for lo, hi in self.frequency_bands:
            mask = (frequencies >= lo) & (frequencies <= hi if hi == self.sampling_hz / 2 else frequencies < hi)
            energy = power[:, mask].sum(1)
            parts += [energy / total, torch.log1p(energy)]
        return torch.nan_to_num(torch.cat(parts, dim=1), nan=0., posinf=1e6, neginf=-1e6)

    @torch.no_grad()
    def fit_feature_normalizer(self, training_x, batch_size=256):
        device = self.feature_mean.device
        total = torch.zeros_like(self.feature_mean, dtype=torch.float64)
        squares = torch.zeros_like(total)
        count = 0
        for start in range(0, len(training_x), batch_size):
            x = torch.as_tensor(training_x[start:start + batch_size], dtype=torch.float32, device=device)
            features = self.extract_features(x).double()
            total += features.sum(0)
            squares += features.square().sum(0)
            count += len(features)
        if not count:
            raise ValueError('Cannot fit feature normalization without source training samples')
        mean = total / count
        self.feature_mean.copy_(mean)
        self.feature_scale.copy_((squares / count - mean.square()).clamp_min(0).sqrt().clamp_min(1e-6))
        self.statistics_fitted.fill_(True)

    def forward(self, x):
        features = self.extract_features(x)
        return self.classifier(((features - self.feature_mean) / self.feature_scale).clamp(-12, 12))


def build_model(input_dim, num_classes, params):
    return FeatureMLP(input_dim, num_classes, params)
