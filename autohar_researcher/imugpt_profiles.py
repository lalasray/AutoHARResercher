from copy import deepcopy


def _profile(sensor_dataset, joint, gyroscope, placement):
    kinds = ('acc', 'gyro') if gyroscope else ('acc',)
    return {'sensor_dataset': sensor_dataset, 'placement': placement,
            'channel_names': ['ax', 'ay', 'az'] + (['gx', 'gy', 'gz'] if gyroscope else []),
            'channel_map': [{'kind': kind, 'joint': joint, 'axis': axis, 'scale': 1.0}
                            for kind in kinds for axis in range(3)]}


DATASET_PROFILES = {
    'wisdm': _profile('usc-had', 'Right_hip', False, 'front trouser pocket approximated by right hip'),
    'uci_har': _profile('realworld', 'Spine1', True, 'waist'),
    'utd_mhad': _profile('pamap', 'Right_wrist', True, 'right wrist; actions 1–21'),
    'hapt': _profile('realworld', 'Spine1', True, 'waist'),
    'usc_had': _profile('usc-had', 'Right_hip', True, 'right hip'),
    'unimib_shar': _profile('usc-had', 'Right_hip', False, 'trouser pocket approximated by right hip'),
    'motion_sense': _profile('usc-had', 'Right_hip', True, 'front trouser pocket approximated by right hip'),
    'real_life_har': _profile('realworld', 'Spine1', True, 'variable carried-phone placement approximated by waist'),
    'hang_time': _profile('pamap', 'Right_wrist', False, 'wrist, using right wrist'),
    'capture_24': _profile('realworld', 'Left_wrist', False, 'wrist, using left wrist'),
}


def dataset_profiles(overrides=None):
    profiles = deepcopy(DATASET_PROFILES)
    for name, override in (overrides or {}).items():
        if name not in profiles:
            raise ValueError(f'Unknown benchmark dataset: {name}')
        profiles[name].update(deepcopy(override))
    return profiles
