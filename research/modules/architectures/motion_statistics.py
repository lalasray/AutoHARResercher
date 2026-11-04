"""Local CNN/BiGRU-attention model with a parallel window-statistics branch.

Default first-convolution stride 2 matches the selected USC-HAD architecture.
custom_params.initial_stride=1 selects the UniMiB temporal-resolution variant.
Checkpoint layer names match the original local implementations.
"""
import math
import torch
from .common import as_sequence

ARCHITECTURE_NAME = 'motion_statistics'


class MotionAndStatistics(torch.nn.Module):
    def __init__(self, channels, classes, params, *, slopes=False, default_stride=2):
        super().__init__()
        self.input_channels = channels
        self.slopes = slopes
        stride = params.get('custom_params', {}).get('initial_stride', default_stride)
        if type(stride) is not int or stride not in (1, 2):
            raise ValueError('initial_stride must be 1 or 2')
        filters = int(params.get('conv_filters', 48))
        second = max(1, filters * 4 // 3)
        hidden = int(params.get('hidden_dim', 48))
        self.features = torch.nn.Sequential(
            torch.nn.Conv1d(3 * channels, filters, 7, stride=stride, padding=3),
            torch.nn.GroupNorm(math.gcd(8, filters), filters), torch.nn.SiLU(),
            torch.nn.Conv1d(filters, second, 7, stride=2, padding=3),
            torch.nn.GroupNorm(math.gcd(8, second), second), torch.nn.SiLU(), torch.nn.Dropout(.2))
        self.gru = torch.nn.GRU(second, hidden, batch_first=True, bidirectional=True)
        self.attention = torch.nn.Linear(2 * hidden, 1)
        self.statistics = torch.nn.Sequential(
            torch.nn.Linear((8 if slopes else 6) * channels, 64),
            torch.nn.SiLU(), torch.nn.Dropout(.2))
        self.head = torch.nn.Sequential(
            torch.nn.Linear(2 * hidden + 64, 64), torch.nn.SiLU(),
            torch.nn.Dropout(params.get('dropout', .3)), torch.nn.Linear(64, classes))

    def forward(self, x):
        x = as_sequence(x, self.input_channels)
        if not x.shape[1] or (self.slopes and x.shape[1] < 4):
            raise ValueError('MotionSlope requires at least four samples; windows must be nonempty')
        mean = x.mean(1)
        std = x.std(1, unbiased=False)
        centered = x - mean[:, None, :]
        quarter = max(1, x.shape[1] // 4)
        parts = [mean, std, x[:, :quarter].mean(1), x[:, -quarter:].mean(1), x.amax(1), x.amin(1)]
        if self.slopes:
            parts.extend([x[:, quarter:2 * quarter].mean(1) - x[:, :quarter].mean(1),
                          x[:, -quarter:].mean(1) - x[:, -2 * quarter:-quarter].mean(1)])
        sequence = self.features(torch.cat([x, centered, centered / std[:, None, :].clamp_min(.1)], 2).transpose(1, 2)).transpose(1, 2)
        sequence, _ = self.gru(sequence)
        temporal = (sequence * self.attention(sequence).softmax(1)).sum(1)
        return self.head(torch.cat([temporal, self.statistics(torch.cat(parts, 1))], 1))


def build_model(input_dim, num_classes, params):
    return MotionAndStatistics(int(params['input_channels']), num_classes, params)
