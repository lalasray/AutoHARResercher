from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np

from .imugpt_profiles import dataset_profiles

UPSTREAM_REVISION = '4b7a73fa7c8b6fdbd18a389b902c0d935a77fc4f'
JOINTS = {
    'realworld': {'Left_knee', 'Left_ankle', 'Head', 'Left_wrist', 'Spine1', 'Spine3', 'Left_elbow'},
    'pamap': {'Right_ankle', 'Right_wrist', 'Spine3'},
    'usc-had': {'Right_hip', 'Right_knee'},
}


class IMUGPTService:
    def __init__(self, config: dict):
        self.config = dict(config)
        root = Path(config['repository']).expanduser().resolve()
        self.config['repository'] = str(root)
        for key in ('vq_checkpoint', 'motion_checkpoint', 'mean', 'std'):
            path = Path(config[key]).expanduser()
            path = path if path.is_absolute() else root / path
            if not path.is_file():
                raise FileNotFoundError(f'Missing IMUGPT asset {key}: {path}. Follow the official setup instructions.')
            self.config[key] = str(path.resolve())
        for name in ('sensor.py', 'models/vqvae.py', 'models/t2m_trans.py'):
            if not (root / name).is_file():
                raise FileNotFoundError(f'Incomplete official IMUGPT checkout: {root / name}')
        self.revision = subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip()
        if self.revision != config.get('revision', UPSTREAM_REVISION):
            raise ValueError('IMUGPT checkout does not match configured revision')
        self.profiles = dataset_profiles(config.get('dataset_profiles'))
        for profile in self.profiles.values():
            self._validate_mapping(profile)
        if 'channel_map' in config:
            self._validate_mapping(config)
        if not np.isfinite(config.get('body_scale', 1.7 / 1.55)) or config.get('body_scale', 1.7 / 1.55) <= 0:
            raise ValueError('body_scale must be finite and positive')

    @staticmethod
    def _validate_mapping(profile):
        mapping = profile['channel_map']
        joints = JOINTS.get(profile['sensor_dataset'], set())
        if not mapping or any(c['kind'] not in ('acc', 'gyro') or c['joint'] not in joints
                              or type(c['axis']) is not int or c['axis'] not in (0, 1, 2)
                              or not np.isfinite(c.get('scale', 1)) or c.get('scale', 1) == 0 for c in mapping):
            raise ValueError('Invalid sensor channel mapping for the upstream sensor dataset')
        names = profile.get('channel_names')
        if names is not None and (len(names) != len(mapping) or len(set(names)) != len(names)):
            raise ValueError('Channel names must uniquely identify every mapped channel')

    def generate(self, label: str, count: int, sampling_hz: float, window_steps: int,
                 channels: int, seed: int, prompt: str, dataset: str | None = None,
                 channel_names: list[str] | None = None) -> dict:
        if not label.strip() or not prompt.strip():
            raise ValueError('Label and motion description are required')
        if type(count) is not int or not 1 <= count <= self.config.get('max_count', 200):
            raise ValueError('Sample count is outside server limits')
        if (type(sampling_hz) not in (int, float) or not np.isfinite(sampling_hz)
                or not 1 <= sampling_hz <= 1000 or type(window_steps) is not int
                or not 0.5 <= window_steps / sampling_hz <= 10):
            raise ValueError('Expected a finite 1–1000 Hz sampling rate and a 0.5–10 second window')
        config = deepcopy(self.config)
        selected = dataset or config.get('default_dataset')
        if selected is not None:
            if selected not in self.profiles:
                raise ValueError(f'Unknown benchmark dataset: {selected}')
            config.update(deepcopy(self.profiles[selected]))
        elif 'channel_map' not in config:
            raise ValueError('A dataset is required when no default sensor mapping is configured')
        if type(channels) is not int or channels != len(config['channel_map']):
            raise ValueError('Requested channels do not match configured physical sensor mapping')
        if channel_names is not None and channel_names != config.get('channel_names'):
            raise ValueError('Requested channel order does not match the dataset sensor profile')
        if type(seed) is not int or not 0 <= seed < 2**32:
            raise ValueError('Seed must be an unsigned 32-bit integer')
        request = dict(label=label, count=count, sampling_hz=sampling_hz,
                       window_steps=window_steps, channels=channels, seed=seed, prompt=prompt, dataset=selected)
        with tempfile.TemporaryDirectory(prefix='autohar-imugpt-') as directory:
            work = Path(directory)
            (work / 'config.json').write_text(json.dumps(config))
            (work / 'request.json').write_text(json.dumps(request))

            subprocess.run([self.config['python'], str(Path(__file__).with_name('imugpt_worker.py')),
                            '--config', str(work / 'config.json'), '--request', str(work / 'request.json'),
                            '--output', str(work / 'samples.npy')], check=True,
                           stdout=sys.stderr, stderr=sys.stderr,
                           timeout=self.config.get('timeout_seconds', 3600))
            samples = np.load(work / 'samples.npy', allow_pickle=False)
        if samples.shape != (count, window_steps, channels) or not np.isfinite(samples).all():
            raise ValueError('IMUGPT worker returned invalid sensor samples')
        return {'samples': samples.tolist(), 'label': label, 'dataset': selected,
                'provenance': {'repository': 'https://github.com/ZikangLeng/IMUGPT',
                               'revision': self.revision, 'seed': seed,
                               'channel_map': config['channel_map'], 'sampling_hz': sampling_hz,
                               'motion_sampling': 'categorical', 'sensor': 'ideal',
                               'calibration': 'none'}}


def main():
    parser = argparse.ArgumentParser(description='IMUGPT text-to-IMU MCP service')
    parser.add_argument('--config', required=True)
    parser.add_argument('--check', action='store_true', help='Validate checkout, assets and channel configuration')
    args = parser.parse_args()
    service = IMUGPTService(json.loads(Path(args.config).read_text()))
    if args.check:
        print(json.dumps({'revision': service.revision, 'datasets': {name: profile['channel_names']
            for name, profile in service.profiles.items()}}))
        return
    try:
        from mcp.server.mcpserver import MCPServer as FastMCP
    except ImportError:
        from mcp.server.fastmcp import FastMCP
    server = FastMCP('IMUGPT')
    server.tool(name='generate_imu')(service.generate)
    server.run(transport='stdio')


if __name__ == '__main__':
    main()
