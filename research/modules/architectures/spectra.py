"""SPECTRA. FFT/hop/width/dropout and the spectral-to-channel
reshape are not specified by the paper. Our explicit interpretation retains C
mixed feature maps in the separable block and projects their frequency bins to
D features before C×C attention. These choices are not reported author settings.
"""
import math
import torch
from torch import nn
from .common import as_sequence


class ChannelAttention(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.query = nn.Linear(width, width, bias=False)
        self.key = nn.Linear(width, width, bias=False)
        self.value = nn.Linear(width, width, bias=False)
        self.gamma = nn.Parameter(torch.zeros(()))

    def forward(self, x):
        weights = torch.softmax(self.query(x) @ self.key(x).transpose(-1, -2) / math.sqrt(x.shape[-1]), dim=-1)
        return x + self.gamma * (weights @ self.value(x))


class SPECTRA(nn.Module):
    def __init__(self, input_dim, num_classes, params):
        super().__init__()
        self.channels = params['input_channels']
        self.n_fft = params.get('n_fft', 32)
        self.hop = params.get('stft_hop', 16)
        if self.n_fft > input_dim // self.channels or self.n_fft < 2 or self.hop < 1:
            raise ValueError('STFT requires n_fft <= window length and a positive hop')
        kernel = params.get('kernel_size', 3)
        if kernel % 2 == 0:
            raise ValueError('SPECTRA temporal convolution kernel must be odd')
        self.register_buffer('hann', torch.hann_window(self.n_fft))
        self.separable = nn.Sequential(
            nn.Conv2d(self.channels, self.channels, (kernel, 1), padding=(kernel // 2, 0), groups=self.channels),
            nn.Conv2d(self.channels, self.channels, 1), nn.BatchNorm2d(self.channels), nn.ReLU())
        width, hidden = params.get('conv_filters', 16), params.get('hidden_dim', 16)
        self.frequency_projection = nn.Linear(self.n_fft // 2 + 1, width)
        self.channel_attention = ChannelAttention(width)
        self.temporal_projection = nn.Linear(self.channels * width, hidden, bias=False)
        self.recurrent = nn.GRU(hidden, hidden, bidirectional=True, batch_first=True)
        self.temporal_score = nn.Linear(hidden * 2, 1, bias=False)
        self.dropout = nn.Dropout(params.get('dropout', .3))
        self.classifier = nn.Linear(hidden * 2, num_classes)

    def stft(self, x):
        batch, time, channels = x.shape
        magnitude = torch.stft(x.transpose(1, 2).reshape(batch * channels, time),
            n_fft=self.n_fft, hop_length=self.hop, window=self.hann,
            center=False, return_complex=True).abs()
        return magnitude.reshape(batch, channels, magnitude.shape[-2], magnitude.shape[-1]).transpose(2, 3)

    def forward(self, x):
        x = self.stft(as_sequence(x, self.channels))  # B,C,L,F
        x = x + self.separable(x)
        x = self.frequency_projection(x.permute(0, 2, 1, 3))  # B,L,C,D
        x = self.channel_attention(x).flatten(2)
        sequence, _ = self.recurrent(self.temporal_projection(x))
        weights = torch.softmax(self.temporal_score(sequence), dim=1)
        return self.classifier(self.dropout((sequence * weights).sum(dim=1)))


def build_model(input_dim, num_classes, params):
    return SPECTRA(input_dim, num_classes, params)
