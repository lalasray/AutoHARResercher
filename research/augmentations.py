"""Source-linked catalog and validation for training-only PyTorch augmentations.

Local implementations/adaptations, not a claim of bit-identical upstream RNG.
All temporal transforms use one shared time mapping across sensor channels.
"""
from __future__ import annotations

import copy
import math
import numpy as np

SURVEY = 'https://github.com/uchidalab/time_series_augmentation/tree/656318d18c0945080c969ce2bf839760880b8d30'
WEARABLE = 'https://github.com/terryum/Data-Augmentation-For-Wearable-Sensor-Data/tree/e5b28e6c884ba5b1fc656f39ba452390cd786b8c'
SPEC = 'https://arxiv.org/abs/1904.08779'
MIXUP = 'https://github.com/facebookresearch/mixup-cifar10'
CUTMIX = 'https://github.com/clovaai/CutMix-PyTorch'

CATALOG = {
    'jitter': dict(defaults={'sigma': .03}, source=SURVEY, description='Add Gaussian noise.'),
    'scaling': dict(defaults={'sigma': .1}, source=SURVEY, description='One Gaussian gain per window/channel.'),
    'rotation': dict(defaults={'max_angle': math.pi / 6}, source=WEARABLE,
        description='Random-axis proper 3D rotation; same rotation for all XYZ triplets. Angle bound is a local conservative default.'),
    'magnitude_warp': dict(defaults={'sigma': .2, 'knots': 4}, source=SURVEY,
        description='Per-channel cubic-spline magnitude envelope.'),
    'time_warp': dict(defaults={'sigma': .2, 'knots': 4}, source=WEARABLE,
        description='Positive spline speed integrated into one monotone time map shared across channels.'),
    'permutation': dict(defaults={'max_segments': 5}, source=SURVEY,
        description='Shuffle equal-length temporal segments, synchronously across channels.'),
    'window_slice': dict(defaults={'ratio': .9}, source=SURVEY,
        description='Random crop then interpolate back to the original window length.'),
    'window_warp': dict(defaults={'ratio': .1, 'scales': [.5, 2.]}, source=SURVEY,
        description='Stretch/compress an interior temporal segment then restore length.'),
    'time_mask': dict(defaults={'ratio': .1}, source=SPEC,
        description='Raw-signal adaptation of time masking; zero one contiguous span across all channels.'),
    'frequency_mask': dict(defaults={'ratio': .1}, source=SPEC,
        description='FFT-domain adaptation, not original spectrogram SpecAugment: zero one contiguous non-DC frequency band, then invert.'),
    'channel_dropout': dict(defaults={'rate': .1}, source='https://jmlr.org/papers/v15/srivastava14a.html',
        description='Input-channel dropout adaptation: zero channels without changing channel order/count; retain at least one channel.'),
    'spawner': dict(defaults={'sigma': .05, 'band_ratio': .1}, source=SURVEY,
        description='Same-class, distinct-window donor; split DTW alignment and averaging, then jitter. Singleton classes remain unchanged.'),
    'random_guided_warp': dict(defaults={'band_ratio': .1}, source=SURVEY,
        description='Same-class DTW-guided resampling using a distinct donor; singleton classes remain unchanged.'),
    'mixup': dict(defaults={'alpha': .2}, source=MIXUP,
        description='Convex mixing of training windows AND class-probability targets; alpha .2 is a local default.'),
    'temporal_cutmix': dict(defaults={'alpha': 1.}, source=CUTMIX,
        description='1D adaptation of CutMix: replace a time span across every channel; target weights use the actual replaced fraction.'),
}
MIXING = {'mixup', 'temporal_cutmix'}
CLASS_AWARE = {'spawner', 'random_guided_warp'}


def catalog():
    return copy.deepcopy(CATALOG)


def _finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def validate_method(spec, mixing_only=False):
    if not isinstance(spec, dict) or set(spec) - {'kind', 'params', 'probability'}:
        raise ValueError('An augmentation step uses kind, params and probability only')
    name = spec.get('kind')
    if name not in CATALOG or (mixing_only and name not in MIXING):
        raise ValueError(f'Unknown augmentation method: {name}')
    probability = spec.get('probability', 1.)
    if not _finite(probability) or not 0 <= probability <= 1:
        raise ValueError('Augmentation probability must be in [0,1]')
    params = spec.get('params', {})
    if not isinstance(params, dict) or set(params) - set(CATALOG[name]['defaults']):
        raise ValueError(f'Unsupported parameters for {name}')
    values = dict(CATALOG[name]['defaults'], **params)
    for key, value in values.items():
        if key == 'scales':
            if not isinstance(value, list) or not value or len(value) > 8 or any(not _finite(v) or not .1 <= v <= 4 for v in value):
                raise ValueError('Warp scales must be a short list in [.1,4]')
        elif key in ('knots', 'max_segments'):
            if type(value) is not int or not 1 <= value <= 16:
                raise ValueError(f'{key} must be an integer in [1,16]')
        elif not _finite(value):
            raise ValueError(f'{key} must be finite')
        elif key in ('sigma', 'max_angle') and value < 0:
            raise ValueError(f'{key} must be nonnegative')
        elif key in ('ratio', 'band_ratio') and not 0 < value <= 1:
            raise ValueError(f'{key} must be in (0,1]')
        elif key == 'rate' and not 0 <= value <= 1:
            raise ValueError('Dropout rate must be in [0,1]')
        elif key == 'alpha' and value <= 0:
            raise ValueError('Mixing alpha must be positive')
    return values


def policy_steps(policy):
    kind = policy.get('kind', 'none')
    if kind == 'compose':
        return copy.deepcopy(policy.get('transforms', []))
    if kind in ('none', 'custom') or kind in MIXING:
        return []
    params = dict(policy.get('params', {}))
    # Single-transform strength is a shorthand; explicit parameters take precedence.
    if kind in ('jitter', 'scaling', 'rotation') and 'strength' in policy:
        params.setdefault('max_angle' if kind == 'rotation' else 'sigma', policy['strength'])
    return [dict(kind=kind, params=params, probability=policy.get('probability', 1.))]


def mixing_spec(policy):
    if policy.get('kind') in MIXING:
        return {k: policy[k] for k in ('kind', 'params', 'probability') if k in policy}
    return policy.get('mixing')


def validate_policy(policy):
    if 'strength' in policy and (not _finite(policy['strength']) or policy['strength'] < 0):
        raise ValueError('Augmentation strength must be finite and nonnegative')
    overrides = policy.get('per_class', {})
    if not isinstance(overrides, dict) or any(not isinstance(label, str) or not label for label in overrides):
        raise ValueError('per_class must map activity names to complete policies')
    if overrides and policy.get('kind') == 'custom':
        raise ValueError('per_class requires a built-in global policy')
    for override in overrides.values():
        allowed = {'kind', 'params', 'probability', 'strength', 'transforms', 'mixing'}
        if not isinstance(override, dict) or set(override) - allowed or 'kind' not in override:
            raise ValueError('Each per_class entry must be a complete non-nested conventional policy with kind')
        if override['kind'] == 'custom':
            raise ValueError('per_class supports built-in conventional methods only')
        validate_policy(override)
    kind = policy.get('kind', 'none')
    if kind not in set(CATALOG) | {'none', 'compose', 'custom'}:
        raise ValueError('Unsupported augmentation kind')
    if kind == 'compose':
        if not isinstance(policy.get('transforms'), list) or not 1 <= len(policy['transforms']) <= 4:
            raise ValueError('Composition requires 1–4 signal transforms')
        if policy.get('params'):
            raise ValueError('Composition parameters belong to individual steps')
    elif policy.get('transforms'):
        raise ValueError('transforms requires kind=compose')
    if kind in ('none', 'custom') and policy.get('params'):
        raise ValueError('none/custom do not consume params')
    if not _finite(policy.get('probability', 1.)) or not 0 <= policy.get('probability', 1.) <= 1:
        raise ValueError('Augmentation probability must be in [0,1]')
    if kind in MIXING and policy.get('mixing'):
        raise ValueError('Configure only one label-mixing method')
    for step in policy_steps(policy):
        validate_method(step)
        if step['kind'] in MIXING:
            raise ValueError('Label mixing belongs in mixing, after signal transforms')
    mix = mixing_spec(policy)
    if mix is not None:
        validate_method(mix, mixing_only=True)


def policy_provenance(policy):
    """Resolved parameters and primary sources persisted with experiment artifacts."""
    steps = policy_steps(policy)
    mix = mixing_spec(policy)
    return {
        'implementation': 'device-resident PyTorch research-derived implementations/adaptations',
        'kind': policy.get('kind', 'none'),
        'probability': policy.get('probability', 1.),
        'per_class': {label: policy_provenance(value) for label, value in policy.get('per_class', {}).items()},
        'signal_steps': [dict(step, params=validate_method(step),
                              source=CATALOG[step['kind']]['source']) for step in steps],
        'mixing': (dict(mix, params=validate_method(mix, mixing_only=True),
                        source=CATALOG[mix['kind']]['source']) if mix else None),
    }


def validate_class_names(policy, class_names):
    unknown = set(policy.get('per_class', {})) - set(class_names)
    if unknown:
        raise ValueError(f'Augmentation classes outside the fixed label space: {sorted(unknown)}')
