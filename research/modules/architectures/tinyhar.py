"""Author TinyHAR implementation with only a [B,T,C] → [B,1,T,C] adapter."""
from torch import nn
from research.vendor.tinyhar import TinyHAR_Model
from .common import as_sequence


class TinyHAR(nn.Module):
    def __init__(self, input_dim, num_classes, params):
        super().__init__()
        self.channels = params['input_channels']
        self.original = TinyHAR_Model((1, 1, input_dim // self.channels, self.channels), num_classes,
            filter_num=params.get('hidden_dim', 20),
            cross_channel_interaction_type='transformer',
            cross_channel_aggregation_type='FC', temporal_info_interaction_type='lstm',
            temporal_info_aggregation_type='tnaive', dropout=params.get('dropout', .1))

    def forward(self, x):
        return self.original(as_sequence(x, self.channels).unsqueeze(1))


def build_model(input_dim, num_classes, params):
    return TinyHAR(input_dim, num_classes, params)
