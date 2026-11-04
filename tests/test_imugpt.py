import json
from pathlib import Path
import sys

import numpy as np
import pytest

from autohar_researcher.imugpt_mcp import IMUGPTService, UPSTREAM_REVISION


@pytest.fixture
def service(tmp_path, monkeypatch):
    config = json.loads(Path('examples/imugpt-server.json').read_text())
    config['repository'] = str(tmp_path)
    config['python'] = sys.executable
    for name in [config[k] for k in ('vq_checkpoint', 'motion_checkpoint', 'mean', 'std')] + ['sensor.py', 'models/vqvae.py', 'models/t2m_trans.py']:
        p = tmp_path / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.touch()
    monkeypatch.setattr('subprocess.check_output', lambda *a, **k: UPSTREAM_REVISION)
    return IMUGPTService(config)


def test_worker_contract_and_provenance(service, monkeypatch):
    def run(command, **kwargs):
        assert kwargs['check'] and kwargs['stdout'] is sys.stderr
        request = json.loads(Path(command[command.index('--request') + 1]).read_text())
        assert request['prompt'] == 'A person walks' and request['seed'] == 37
        np.save(command[command.index('--output') + 1], np.ones((2, 128, 6), dtype=np.float32))
    monkeypatch.setattr('subprocess.run', run)
    result = service.generate('walking', 2, 50, 128, 6, 37, 'A person walks')
    assert np.asarray(result['samples']).shape == (2, 128, 6)
    assert result['provenance']['revision'] == UPSTREAM_REVISION


@pytest.mark.parametrize('field,value', [('count', 0), ('count', 201), ('channels', 3), ('sampling_hz', 0), ('sampling_hz', float('nan')), ('window_steps', 501), ('dataset', 'unknown'), ('channel_names', ['az', 'ay', 'ax']), ('seed', -1), ('prompt', '')])
def test_reject_invalid_requests_before_worker(service, monkeypatch, field, value):
    def unexpected(*a, **k):
        pytest.fail('Invalid request launched worker')
    monkeypatch.setattr('subprocess.run', unexpected)
    args = dict(label='walking', count=2, sampling_hz=50, window_steps=128, channels=6, seed=37, prompt='walk')
    args[field] = value
    with pytest.raises(ValueError):
        service.generate(**args)


def test_worker_bad_output_fails(service, monkeypatch):
    def run(command, **kwargs):
        np.save(command[command.index('--output') + 1], np.zeros((1, 128, 3)))
    monkeypatch.setattr('subprocess.run', run)
    with pytest.raises(ValueError, match='invalid sensor'):
        service.generate('walking', 1, 50, 128, 6, 37, 'walk')


def test_missing_assets_and_wrong_revision(service, monkeypatch):
    monkeypatch.setattr('subprocess.check_output', lambda *a, **k: 'wrong')
    with pytest.raises(ValueError, match='revision'):
        IMUGPTService(service.config)
    Path(service.config['mean']).unlink()
    with pytest.raises(FileNotFoundError, match='Missing IMUGPT asset'):
        IMUGPTService(service.config)


def test_imugpt_server_over_real_mcp_stdio(service, tmp_path, monkeypatch):
    pytest.importorskip('mcp')
    import subprocess
    from autohar_researcher.synthetic import MCPGenerator
    monkeypatch.undo()
    subprocess.run(['git', 'init', '-q', str(tmp_path)], check=True)
    subprocess.run(['git', '-C', str(tmp_path), '-c', 'user.name=Test', '-c', 'user.email=test@example.org',
                    'commit', '--allow-empty', '-qm', 'fixture'], check=True)
    service.config['revision'] = subprocess.check_output(['git', '-C', str(tmp_path), 'rev-parse', 'HEAD'], text=True).strip()
    worker = tmp_path / 'fixture-python'
    worker.write_text('#!' + sys.executable + '\n' + '''import json, sys
from pathlib import Path
import numpy as np
request = json.loads(Path(sys.argv[sys.argv.index('--request') + 1]).read_text())
np.save(sys.argv[sys.argv.index('--output') + 1], np.full((request['count'], request['window_steps'], request['channels']), 1.25, dtype=np.float32))
print('worker diagnostic')
''')
    worker.chmod(0o755)
    service.config['python'] = str(worker)
    config = tmp_path / 'server.json'
    config.write_text(json.dumps(service.config))
    generator = MCPGenerator({'text': {'command': sys.executable, 'args': ['-m', 'autohar_researcher.imugpt_mcp', '--config', str(config)], 'tool': 'generate_imu'}})
    result = generator.generate({'generator': 'text', 'label': 'walking', 'count': 2, 'prompt': 'a person walks'}, 128, 6)
    assert result.shape == (2, 128, 6)
    assert np.all(result == 1.25)


@pytest.mark.parametrize('dataset,joint,channels', [
    ('wisdm', 'Right_hip', 3), ('uci_har', 'Spine1', 6),
    ('utd_mhad', 'Right_wrist', 6), ('hapt', 'Spine1', 6),
    ('usc_had', 'Right_hip', 6), ('unimib_shar', 'Right_hip', 3),
    ('motion_sense', 'Right_hip', 6), ('real_life_har', 'Spine1', 6),
    ('hang_time', 'Right_wrist', 3), ('capture_24', 'Left_wrist', 3),
])
def test_dataset_worker_mapping(service, monkeypatch, dataset, joint, channels):
    original = json.dumps(service.config, sort_keys=True)
    names = ['ax', 'ay', 'az'] + (['gx', 'gy', 'gz'] if channels == 6 else [])
    def run(command, **kwargs):
        config = json.loads(Path(command[command.index('--config') + 1]).read_text())
        request = json.loads(Path(command[command.index('--request') + 1]).read_text())
        assert request['dataset'] == dataset
        assert config['channel_names'] == names
        assert [c['joint'] for c in config['channel_map']] == [joint] * channels
        assert [c['axis'] for c in config['channel_map']] == list(range(3)) * (channels // 3)
        assert [c['kind'] for c in config['channel_map']] == ['acc'] * 3 + (['gyro'] * 3 if channels == 6 else [])
        np.save(command[command.index('--output') + 1], np.zeros((200, 100, channels)))
    monkeypatch.setattr('subprocess.run', run)
    result = service.generate('activity', 200, 20, 100, channels, 1, 'a person moves',
                              dataset=dataset, channel_names=names)
    assert np.asarray(result['samples']).shape == (200, 100, channels)
    assert json.dumps(service.config, sort_keys=True) == original


def test_profiles_cover_benchmark():
    from autohar_researcher.imugpt_profiles import DATASET_PROFILES
    from autohar_researcher.types import LOSO_DATASETS
    assert set(DATASET_PROFILES) == set(LOSO_DATASETS)


def test_dataset_aware_client_routing(monkeypatch):
    from autohar_researcher.synthetic import MCPGenerator
    async def call(self, spec, args):
        assert args['dataset'] == 'capture_24'
        assert args['channel_names'] == ['ax', 'ay', 'az']
        assert args['sampling_hz'] == 100
        return {'samples': np.zeros((2, 100, 3)).tolist()}
    monkeypatch.setattr(MCPGenerator, '_call', call)
    generator = MCPGenerator({'text': {'dataset_profiles': True}})
    request = dict(generator='text', label='walking', count=2, prompt='walk',
                   source_dataset='capture_24', channel_names=['ax', 'ay', 'az'])
    assert generator.generate(request, 100, 3, sampling_hz=100).shape == (2, 100, 3)
    del request['source_dataset']
    with pytest.raises(ValueError, match='source_dataset'):
        generator.generate(request, 100, 3)
