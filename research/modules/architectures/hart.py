"""PyTorch port of the released HART architecture."""
from .torch_author import HART


def build_model(input_dim,num_classes,params):
    if params['input_channels']!=6:
        raise ValueError('Original HART requires acceleration XYZ then gyroscope XYZ')
    return HART(input_dim//6,num_classes,params.get('hidden_dim',192),params.get('dropout',.3))
