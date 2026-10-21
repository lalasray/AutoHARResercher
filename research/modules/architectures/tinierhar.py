"""Author TinierHAR implementation, including residual Conv2d and bidirectional GRU."""
from torch import nn
from research.vendor.tinierhar import TinierHAR_Model
from .common import as_sequence


class TinierHAR(nn.Module):
    def __init__(self, input_dim, num_classes, params):
        super().__init__()
        self.channels = params['input_channels']
        self.original = TinierHAR_Model((1, 1, input_dim // self.channels, self.channels), num_classes, 1,
            {'nb_conv_blocks': params.get('depth', 4), 'nb_units_gru': params.get('hidden_dim', 16),
             'nb_filters': params.get('conv_filters', 4), 'drop_prob': params.get('dropout', .3)})

    def forward(self, x):
        return self.original(as_sequence(x, self.channels).unsqueeze(1))


def build_model(input_dim, num_classes, params):
    return TinierHAR(input_dim, num_classes, params)
