import math
from copy import deepcopy

DEFAULTS = {
    'contratsc': dict(pretrain_epochs=40, teacher_epochs=40, class_aware_epochs=40,
                     temporal_steps=6, temperature=.2, label_fraction=1.,
                     scaling_sigma=1.1, jitter_sigma=.8, max_segments=8),
    'sdmix': dict(bottleneck=256, mixup_alpha=.2, margin=10000., radius='mean',
                  radius_norm=2, radius_update_epochs=1, top_k=1),
    'crosshar': dict(reconstruction_epochs=800, contrastive_epochs=800, mask_ratio=.15,
                     max_span=10, reconstruction_weight=6., temperature=.2,
                     label_fraction=1., scaling_sigma=1.1, jitter_sigma=.01, max_segments=8),
}


def settings(method, overrides):
    if not isinstance(overrides, dict) or set(overrides) - set(DEFAULTS[method]):
        raise ValueError('Unsupported comparison method settings')
    result = dict(deepcopy(DEFAULTS[method]), **overrides)
    for key, value in result.items():
        if key == 'radius':
            if value not in ('mean','max'):
                raise ValueError('radius must be mean or max')
        elif key == 'radius_norm':
            if value not in (1,2):
                raise ValueError('radius_norm must be 1 or 2')
        elif key.endswith('epochs') or key in ('bottleneck','temporal_steps','max_segments','max_span','top_k'):
            if type(value) is not int or value < 1:
                raise ValueError(f'{key} must be a positive integer')
        elif type(value) not in (float,int) or not math.isfinite(value) or value <= 0:
            raise ValueError(f'{key} must be finite and positive')
    for key in ('mask_ratio','label_fraction'):
        if result.get(key,1) > 1:
            raise ValueError(f'{key} must be at most one')
    return result
