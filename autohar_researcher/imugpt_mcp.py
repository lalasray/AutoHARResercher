"""Stdio MCP service for the official IMUGPT text-to-motion-to-IMU pipeline."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np

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
        mapping = config['channel_map']
        joints = JOINTS.get(config['sensor_dataset'], set())
        if not mapping or any(c['kind'] not in ('acc', 'gyro') or c['joint'] not in joints
                              or type(c['axis']) is not int or c['axis'] not in (0, 1, 2)
                              or not np.isfinite(c.get('scale', 1)) or c.get('scale', 1) == 0 for c in mapping):
            raise ValueError('Invalid sensor channel mapping for the upstream sensor dataset')
        if not np.isfinite(config.get('body_scale', 1.7 / 1.55)) or config.get('body_scale', 1.7 / 1.55) <= 0:
            raise ValueError('body_scale must be finite and positive')

    def generate(self, label: str, count: int, sampling_hz: int, window_steps: int,
                 channels: int, seed: int, prompt: str) -> dict:
        if not label.strip() or not prompt.strip():
            raise ValueError('Label and motion description are required')
        if type(count) is not int or not 1 <= count <= self.config.get('max_count', 100):
            raise ValueError('Sample count is outside server limits')
        if sampling_hz != 50 or type(window_steps) is not int or not 50 <= window_steps <= 250:
            raise ValueError('Expected 50 Hz and a 1–5 second window')
        if channels != len(self.config['channel_map']):
            raise ValueError('Requested channels do not match configured physical sensor mapping')
        if type(seed) is not int or not 0 <= seed < 2**32:
            raise ValueError('Seed must be an unsigned 32-bit integer')
        request = dict(label=label, count=count, sampling_hz=sampling_hz,
                       window_steps=window_steps, channels=channels, seed=seed, prompt=prompt)
        with tempfile.TemporaryDirectory(prefix='autohar-imugpt-') as directory:
            work = Path(directory)
            (work / 'config.json').write_text(json.dumps(self.config))
            (work / 'request.json').write_text(json.dumps(request))
            # Upstream progress output must never corrupt MCP stdout.
            subprocess.run([self.config['python'], str(Path(__file__).with_name('imugpt_worker.py')),
                            '--config', str(work / 'config.json'), '--request', str(work / 'request.json'),
                            '--output', str(work / 'samples.npy')], check=True,
                           stdout=sys.stderr, stderr=sys.stderr,
                           timeout=self.config.get('timeout_seconds', 3600))
            samples = np.load(work / 'samples.npy', allow_pickle=False)
        if samples.shape != (count, window_steps, channels) or not np.isfinite(samples).all():
            raise ValueError('IMUGPT worker returned invalid sensor samples')
        return {'samples': samples.tolist(), 'label': label,
                'provenance': {'repository': 'https://github.com/ZikangLeng/IMUGPT',
                               'revision': self.revision, 'seed': seed,
                               'channel_map': self.config['channel_map'], 'sampling_hz': sampling_hz,
                               'motion_sampling': 'categorical', 'sensor': 'ideal',
                               'calibration': 'none'}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--check', action='store_true', help='Validate checkout, assets and channel configuration')
    args = parser.parse_args()
    service = IMUGPTService(json.loads(Path(args.config).read_text()))
    if args.check:
        print(json.dumps({'revision': service.revision, 'channels': len(service.config['channel_map'])}))
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
