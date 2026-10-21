"""TECO TinyHAR benchmark's DeepConvLSTM-attention implementation.

This is explicitly the TECO benchmark variant, not an assertion that it is
Murahari & Ploetz's original implementation.
"""
from torch import nn
from research.vendor.deepconvlstmattention import DeepConvLSTM_ATTN
from .common import as_sequence


class DeepConvLSTMAttention(nn.Module):
    def __init__(self, input_dim, num_classes, params):
        super().__init__()
        self.channels = params['input_channels']
        self.original = DeepConvLSTM_ATTN((1, 1, input_dim // self.channels, self.channels), num_classes, 1,
            {'nb_conv_blocks': 2, 'nb_filters': params.get('conv_filters', 64),
             'dilation': 1, 'batch_norm': 0, 'filter_width': 5,
             'nb_layers_lstm': params.get('lstm_layers', 2),
             'nb_units_lstm': params.get('hidden_dim', 128), 'drop_prob': params.get('dropout', .5)})

    def forward(self, x):
        return self.original(as_sequence(x, self.channels).unsqueeze(1))


def build_model(input_dim, num_classes, params):
    return DeepConvLSTMAttention(input_dim, num_classes, params)
