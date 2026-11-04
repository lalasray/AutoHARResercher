from fractions import Fraction

import numpy as np
from scipy.signal import resample_poly, savgol_filter
from scipy.spatial.transform import Rotation


JOINTS = {'Spine1', 'Right_hip', 'Left_wrist', 'Right_wrist'}


def pose_signal_provenance(profile, coordinate_mode='hip-relative', *, diagnostic_only=False):
    if coordinate_mode not in ('hip-relative', 'absolute-world'):
        raise ValueError('Expected hip-relative or explicitly supplied absolute-world coordinates')
    hip_relative = coordinate_mode == 'hip-relative'
    has_acc = any(c['kind'] == 'acc' for c in profile['channel_map'])
    return dict(schema_version=1, coordinate_mode=coordinate_mode,
        diagnostic_only=diagnostic_only,
        physically_complete_imu=False if hip_relative else None,
        training_eligible=False if hip_relative or diagnostic_only else None,
        common_body_translation_available=not hip_relative,
        root_acceleration_recoverable=not hip_relative,
        sensor_joints=sorted({c['joint'] for c in profile['channel_map']}),
        missing_components=(['common-body translational acceleration'] if hip_relative and has_acc else []),
        waist_origin_degenerate=hip_relative and any(c['kind']=='acc' and c['joint']=='Spine1' for c in profile['channel_map']),
        limitations=['Pose-derived virtual sensor frames and configured gravity are assumptions, not sensor calibration.',
            'Camera rotation/motion must be accounted for by an externally calibrated world trajectory.',
            *(['Hip centering removes whole-body translation; image-plane hip displacement does not supply calibrated 3D root acceleration.'] if hip_relative else
              ['Absolute metric inertial-world coordinates are a caller-supplied assumption, not recovered or certified by this converter.'])])


def validate_pose_generation(profile, *, coordinate_mode='hip-relative', diagnostic_only=False, return_provenance=False):
    if type(diagnostic_only) is not bool:
        raise ValueError('diagnostic_only must be a boolean')
    provenance = pose_signal_provenance(profile, coordinate_mode, diagnostic_only=diagnostic_only)
    if coordinate_mode == 'hip-relative' and (not diagnostic_only or not return_provenance):
        raise ValueError('Hip-relative MediaPipe poses lack whole-body translation and cannot supply complete IMU training signals. '
                         'Spine1 hip-origin acceleration cannot recover body translation. Use explicit diagnostic_only with provenance, '
                         'or a calibrated absolute-motion external adapter.')
    return provenance


def required_landmarks(joint):
    if joint not in JOINTS:
        raise ValueError(f'Unsupported virtual sensor joint: {joint}')
    return [11, 12, 23, 24] + ({'Left_wrist': [13, 15, 17, 19],
                               'Right_wrist': [14, 16, 18, 20]}.get(joint, []))


def _unit(v):
    norm = np.linalg.norm(v, axis=-1, keepdims=True)
    if np.any(norm < 1e-6):
        raise ValueError('Degenerate pose cannot define sensor orientation')
    return v / norm


def sensor_frame(pose, joint):
    hip = (pose[:, 23] + pose[:, 24]) / 2
    shoulder = (pose[:, 11] + pose[:, 12]) / 2
    if joint in ('Spine1', 'Right_hip'):
        position = hip if joint == 'Spine1' else pose[:, 24]
        x = _unit(pose[:, 24] - pose[:, 23])
        guide = shoulder - hip
    else:
        elbow, wrist, pinky, index = (13, 15, 17, 19) if joint == 'Left_wrist' else (14, 16, 18, 20)
        position = pose[:, wrist]
        x = _unit(pose[:, wrist] - pose[:, elbow])
        guide = pose[:, index] - pose[:, pinky]
    z = _unit(np.cross(x, guide))
    y = _unit(np.cross(z, x))
    return position, np.stack((x, y, z), axis=-1)


def pose_to_imu(pose, fps, sampling_hz, profile, *, gravity=(0, 9.80665, 0), smoothing_seconds=.2,
                coordinate_mode='hip-relative', diagnostic_only=False, return_provenance=False):
    provenance = validate_pose_generation(profile, coordinate_mode=coordinate_mode,
        diagnostic_only=diagnostic_only, return_provenance=return_provenance)
    pose = np.asarray(pose, dtype=np.float64)
    if pose.ndim != 3 or pose.shape[1:] != (33, 3) or not np.isfinite(pose).all():
        raise ValueError('Expected finite [frames,33,3] metric poses')
    if not np.isfinite([fps, sampling_hz, smoothing_seconds]).all() or min(fps, sampling_hz) <= 0 or smoothing_seconds <= 0:
        raise ValueError('Invalid signal sampling or smoothing settings')
    width = max(5, int(round(smoothing_seconds * fps)) | 1)
    if len(pose) < 2 * width + 3:
        raise ValueError('Pose segment too short for smoothing and derivatives')
    gravity = np.asarray(gravity, dtype=float)
    if gravity.shape != (3,) or not np.isfinite(gravity).all():
        raise ValueError('Gravity must be a finite reference-frame vector')
    if coordinate_mode == 'hip-relative':
        # Enforce the documented moving origin; do not mistake estimation drift
        # or a residual hip offset for a metric root-motion trajectory.
        pose = pose - ((pose[:, 23] + pose[:, 24]) / 2)[:, None, :]
    smooth = savgol_filter(pose, width, 3, axis=0)
    sensors = {}
    for joint in {c['joint'] for c in profile['channel_map']}:
        required_landmarks(joint)
        position, frame = sensor_frame(smooth, joint)
        acceleration = savgol_filter(position, width, 3, deriv=2, delta=1 / fps, axis=0)
        acceleration = np.einsum('nji,nj->ni', frame, acceleration - gravity)

        delta = np.einsum('nji,njk->nik', frame[:-1], frame[1:])
        angular = Rotation.from_matrix(delta).as_rotvec() * fps
        angular = np.vstack((angular, angular[-1]))
        sensors[joint] = {'acc': acceleration, 'gyro': angular}
    signal = np.stack([sensors[c['joint']][c['kind']][:, c['axis']] * c.get('scale', 1.)
                       for c in profile['channel_map']], axis=1)
    signal = signal[width:-width]
    ratio = Fraction(float(sampling_hz) / fps).limit_denominator(10000)
    signal = resample_poly(signal, ratio.numerator, ratio.denominator, axis=0)
    if not np.isfinite(signal).all():
        raise ValueError('Virtual sensor conversion produced nonfinite values')
    signal = signal.astype(np.float32)
    return {'samples': signal, 'provenance': provenance} if return_provenance else signal
