from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import re
import subprocess
import sys
import tempfile

import numpy as np

from .imugpt_profiles import dataset_profiles
from .imugpt_mcp import IMUGPTService
from .imutube_signal import required_landmarks, validate_pose_generation, pose_signal_provenance


class IMUTubeService:
    def __init__(self, config):
        self.config = deepcopy(config)
        self.backend = config.get('backend', 'mediapipe')
        if type(config.get('diagnostic_only', False)) is not bool:
            raise ValueError('diagnostic_only must be a boolean')
        if self.backend not in ('mediapipe', 'external'):
            raise ValueError('backend must be mediapipe or external')
        if self.backend == 'external':
            command = config.get('external_command')
            if not isinstance(command, list) or not command or any(
                    not isinstance(part, str) or not part.strip() for part in command):
                raise ValueError('external_command must be a nonempty argument list')
        else:
            model = Path(config['pose_model']).expanduser().resolve()
            if not model.is_file():
                raise FileNotFoundError(f'Missing MediaPipe Pose Landmarker model: {model}')
            self.config['pose_model'] = str(model)
        self.profiles = dataset_profiles(config.get('dataset_profiles'))
        for profile in self.profiles.values():
            IMUGPTService._validate_mapping(profile)
            if self.backend == 'mediapipe':
                for channel in profile['channel_map']:
                    required_landmarks(channel['joint'])
        pool = config.get('video_pool', {})
        if not isinstance(pool, dict) or not pool:
            raise ValueError('Configure a nonempty curated video_pool keyed by video ID')
        self.pool = {}
        for identifier, item in pool.items():
            path = Path(item['path']).expanduser().resolve()
            if not path.is_file():
                raise FileNotFoundError(f'Missing curated video {identifier}: {path}')
            if not isinstance(item.get('labels'), list) or not item['labels'] or any(not isinstance(v, str) or not v for v in item['labels']):
                raise ValueError('Each video requires its curated activity labels')
            self.pool[identifier] = dict(item, path=str(path))
        confidence = config.get('confidence', .7)
        if not isinstance(confidence, (int, float)) or not 0 < confidence <= 1:
            raise ValueError('confidence must be in (0,1]')
        for key, default in [('smoothing_seconds', .2), ('max_center_jump', .15), ('timeout_seconds', 3600)]:
            value = config.get(key, default)
            if not isinstance(value, (float, int)) or not np.isfinite(value) or value <= 0:
                raise ValueError(f'{key} must be finite and positive')
        for key, default in [('max_count', 200), ('max_frames', 54000)]:
            value = config.get(key, default)
            if type(value) is not int or value < 1:
                raise ValueError(f'{key} must be a positive integer')
        gravity = np.asarray(config.get('gravity', [0, 9.80665, 0]), dtype=float)
        if gravity.shape != (3,) or not np.isfinite(gravity).all():
            raise ValueError('gravity must be a finite 3-vector')

    def generate(self, label: str, count: int, sampling_hz: float, window_steps: int,
                 channels: int, seed: int, videos: list[str], selection: str = 'all',
                 dataset: str | None = None, channel_names: list[str] | None = None) -> dict:
        if not isinstance(label, str) or not label.strip():
            raise ValueError('An activity label is required')
        if type(count) is not int or not 1 <= count <= self.config.get('max_count', 200):
            raise ValueError('Sample count outside server limits')
        if (type(sampling_hz) not in (float, int) or not np.isfinite(sampling_hz)
                or not 1 <= sampling_hz <= 1000 or type(window_steps) is not int
                or not .5 <= window_steps / sampling_hz <= 10):
            raise ValueError('Expected 1–1000 Hz and a 0.5–10 second window')
        if type(seed) is not int or not 0 <= seed < 2**32:
            raise ValueError('Seed must be an unsigned 32-bit integer')
        dataset = dataset or self.config.get('default_dataset')
        if dataset not in self.profiles:
            raise ValueError('A supported dataset is required')
        profile = deepcopy(self.profiles[dataset])
        if type(channels) is not int or channels != len(profile['channel_map']):
            raise ValueError('Channels do not match dataset profile')
        if channel_names is not None and channel_names != profile['channel_names']:
            raise ValueError('Channel order does not match dataset profile')
        if self.backend == 'mediapipe':
            physics = validate_pose_generation(profile, diagnostic_only=self.config.get('diagnostic_only', False), return_provenance=True)
        else:
            physics = dict(schema_version=1, coordinate_mode=self.config.get('coordinate_mode', 'unspecified'),
                physically_complete_imu=None, diagnostic_only=self.config.get('diagnostic_only', False),
                training_eligible=False if self.config.get('diagnostic_only', False) else None,
                limitations=['External adapter motion recovery and calibration are not verified by this service.'])
        if not isinstance(videos, list) or not videos or any(not isinstance(v, str) for v in videos) or len(set(videos)) != len(videos):
            raise ValueError('videos must contain unique curated video IDs')
        if any(v not in self.pool for v in videos):
            raise ValueError('Video is outside the configured source pool')
        if any(label not in self.pool[v]['labels'] for v in videos):
            raise ValueError('Video label does not match the curated activity')
        selected = list(videos)
        if selection != 'all':
            match = re.fullmatch(r'random\(([1-9][0-9]*)\)', selection)
            if not match or int(match[1]) > len(selected):
                raise ValueError('selection must be all or random(k) within the requested pool')
            selected = np.random.default_rng(seed).choice(selected, int(match[1]), replace=False).tolist()
        request = dict(label=label, count=count, sampling_hz=sampling_hz, window_steps=window_steps,
                       channels=channels, seed=seed, profile=profile, dataset=dataset,
                       channel_names=profile['channel_names'], protocol_version=1,
                       units={'acc': 'm/s^2', 'gyro': 'rad/s'},
                       video_paths=[self.pool[v]['path'] for v in selected])
        with tempfile.TemporaryDirectory(prefix='autohar-video-') as directory:
            directory = Path(directory)
            (directory / 'config.json').write_text(json.dumps(self.config))
            (directory / 'request.json').write_text(json.dumps(request))
            command = (self.config['external_command'] if self.backend == 'external' else
                       [self.config.get('python', sys.executable), '-m', 'autohar_researcher.imutube_worker'])
            subprocess.run([*command, '--config', str(directory / 'config.json'), '--request', str(directory / 'request.json'),
                '--output', str(directory / 'samples.npy')], check=True, stdout=sys.stderr, stderr=sys.stderr,
                timeout=self.config.get('timeout_seconds', 3600))
            samples = np.load(directory / 'samples.npy', allow_pickle=False)
        if (samples.dtype.kind not in 'fi' or samples.shape != (count, window_steps, channels)
                or not np.isfinite(samples).all()):
            raise ValueError('Video worker returned invalid samples')
        return {'samples': samples.tolist(), 'label': label, 'dataset': dataset,
                'training_eligible': physics['training_eligible'], 'provenance': physics,
                'selected_videos': selected, 'implementation': self.config.get('implementation', 'external-imutube') if self.backend == 'external' else 'imutube-inspired-mediapipe',
                'coordinate_mode': self.config.get('coordinate_mode', 'unspecified') if self.backend == 'external' else 'hip-relative',
                'calibration': self.config.get('calibration', 'unspecified') if self.backend == 'external' else 'none'}


def main():
    parser = argparse.ArgumentParser(description='Video-to-IMU MCP service')
    parser.add_argument('--config', required=True)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    path = Path(args.config).resolve()
    config = json.loads(path.read_text())

    for item, key in ([(config, 'pose_model')] if config.get('backend', 'mediapipe') == 'mediapipe' else []) + [(v, 'path') for v in config.get('video_pool', {}).values()]:
        value = Path(item[key]).expanduser()
        item[key] = str(value if value.is_absolute() else path.parent / value)
    service = IMUTubeService(config)
    if args.check:
        print(json.dumps({'backend': service.backend,
                          'datasets': list(service.profiles), 'videos': list(service.pool),
                          'signal_provenance': ({name: pose_signal_provenance(profile,
                              diagnostic_only=config.get('diagnostic_only', False)) for name, profile in service.profiles.items()}
                              if service.backend == 'mediapipe' else {'calibration': 'external_adapter_unverified'})}))
        return
    try:
        from mcp.server.mcpserver import MCPServer as FastMCP
    except ImportError:
        from mcp.server.fastmcp import FastMCP
    server = FastMCP('Video-to-IMU')
    server.tool(name='generate_imu')(service.generate)
    server.run(transport='stdio')


if __name__ == '__main__':
    main()
