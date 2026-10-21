"""PyTorch port of the released MobileHART-XS architecture."""
from .torch_author import MobileHART


def build_model(input_dim,num_classes,params):
    if params['input_channels']!=6 or input_dim//6!=128:
        raise ValueError('Original MobileHART-XS requires 128 samples and acc XYZ + gyro XYZ')
    if params.get('hidden_dim',144)!=144:
        raise ValueError('Original MobileHART-XS uses projection widths96/120/144')
    return MobileHART(num_classes,params.get('dropout',.3))
