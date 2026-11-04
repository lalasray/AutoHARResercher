import json
from pathlib import Path
import sys

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from autohar_researcher.imugpt_profiles import DATASET_PROFILES
from autohar_researcher.imutube_mcp import IMUTubeService
from autohar_researcher.imutube_signal import pose_to_imu
from autohar_researcher.imutube_worker import generate


def poses(n=180):
    p = np.zeros((n, 33, 3))
    for i, xyz in {23: [-.2, 0, 0], 24: [.2, 0, 0], 11: [-.2, -1, 0], 12: [.2, -1, 0],
                   13: [-.4, -.6, 0], 15: [-.7, -.6, 0], 17: [-.8, -.65, 0], 19: [-.8, -.55, 0],
                   14: [.4, -.6, 0], 16: [.7, -.6, 0], 18: [.8, -.65, 0], 20: [.8, -.55, 0]}.items():
        p[:, i] = xyz
    return p


@pytest.mark.parametrize('dataset', list(DATASET_PROFILES))
def test_stationary_sensor_has_gravity_and_no_rotation(dataset):
    signal = pose_to_imu(poses(), 30, 50, DATASET_PROFILES[dataset], coordinate_mode='absolute-world')
    assert signal.shape[1] == len(DATASET_PROFILES[dataset]['channel_names'])
    np.testing.assert_allclose(np.linalg.norm(signal[20:-20, :3], axis=1), 9.80665, atol=.01)
    if signal.shape[1] == 6:
        np.testing.assert_allclose(signal[:, 3:], 0, atol=1e-6)


def test_acceleration_and_angular_velocity_units():
    p = poses()
    t = np.arange(len(p)) / 30
    p[:, :, 0] += t[:, None] ** 2
    result = pose_to_imu(p, 30, 30, DATASET_PROFILES['uci_har'], gravity=[0, 0, 0], coordinate_mode='absolute-world')
    np.testing.assert_allclose(result[:, 0], 2, atol=1e-5)
    rotations = Rotation.from_rotvec(np.column_stack([t * 0, t * 0, .5 * t])).as_matrix()
    p = np.einsum('nij,nkj->nki', rotations, poses())
    result = pose_to_imu(p, 30, 30, DATASET_PROFILES['uci_har'], gravity=[0, 0, 0], coordinate_mode='absolute-world')
    np.testing.assert_allclose(np.linalg.norm(result[:, 3:], axis=1), .5, atol=1e-3)


@pytest.fixture
def config(tmp_path):
    (tmp_path / 'pose.task').touch()
    pool = {}
    for name in ['one', 'two']:
        (tmp_path / (name + '.mp4')).touch()
        pool[name] = {'path': str(tmp_path / (name + '.mp4')), 'labels': ['walking']}
    return {'pose_model': str(tmp_path / 'pose.task'), 'video_pool': pool, 'python': sys.executable,
            'diagnostic_only': True}


def arguments():
    return dict(label='walking', count=2, sampling_hz=50, window_steps=50, channels=3,
                seed=13, videos=['one', 'two'], dataset='capture_24', channel_names=['ax', 'ay', 'az'])


def test_service_selection_and_worker_contract(config, monkeypatch):
    calls = []
    def run(command, **kwargs):
        request = json.loads(Path(command[command.index('--request') + 1]).read_text())
        calls.append(request)
        assert kwargs['stdout'] is sys.stderr and kwargs['check']
        assert request['profile']['channel_map'][0]['joint'] == 'Left_wrist'
        np.save(command[command.index('--output') + 1], np.ones((2, 50, 3)))
    monkeypatch.setattr('subprocess.run', run)
    service = IMUTubeService(config)
    a = service.generate(**arguments(), selection='random(1)')
    b = service.generate(**arguments(), selection='random(1)')
    assert a['selected_videos'] == b['selected_videos']
    assert len(calls[0]['video_paths']) == 1
    assert np.asarray(a['samples']).shape == (2, 50, 3)
    assert a['training_eligible'] is False
    assert a['provenance']['coordinate_mode'] == 'hip-relative'
    assert a['provenance']['physically_complete_imu'] is False
    assert 'common-body translational acceleration' in a['provenance']['missing_components']


@pytest.mark.parametrize('key,value', [('label','running'), ('videos',['outside']), ('videos',['one','one']),
    ('selection','random(3)'), ('selection','random(0)'), ('count',201), ('sampling_hz',float('nan')),
    ('window_steps',1), ('channels',6), ('dataset','unknown'), ('seed',-1), ('channel_names',['az','ay','ax'])])
def test_invalid_request_does_not_launch_worker(config, monkeypatch, key, value):
    monkeypatch.setattr('subprocess.run', lambda *a, **k: pytest.fail('Worker launched'))
    args = arguments()
    args[key] = value
    with pytest.raises(ValueError):
        IMUTubeService(config).generate(**args)


def test_worker_keeps_tracking_segments_separate(monkeypatch):

    monkeypatch.setattr('autohar_researcher.imutube_worker.extract_segments',
                        lambda *args: iter([(poses(30), 30), (poses(30), 30)]))
    request = dict(seed=13, video_paths=['one'], profile=DATASET_PROFILES['uci_har'],
                   sampling_hz=30, window_steps=30, count=1)
    with pytest.raises(ValueError, match='valid window'):
        generate({'diagnostic_only': True}, request)


def test_worker_deterministic_windows_and_insufficient_count(monkeypatch):
    monkeypatch.setattr('autohar_researcher.imutube_worker.extract_segments',
                        lambda *args: iter([(poses(), 30)]))
    request = dict(seed=13, video_paths=['one'], profile=DATASET_PROFILES['uci_har'],
                   sampling_hz=50, window_steps=50, count=2)
    np.testing.assert_array_equal(generate({'diagnostic_only': True}, request), generate({'diagnostic_only': True}, request))
    request['count'] = 1000
    with pytest.raises(ValueError, match='valid window'):
        generate({'diagnostic_only': True}, request)


def test_video_client_dataset_routing(monkeypatch):
    from autohar_researcher.synthetic import MCPGenerator
    async def call(self, spec, args):
        assert args['dataset'] == 'capture_24'
        assert args['channel_names'] == ['ax','ay','az']
        assert args['videos'] == ['one'] and args['selection'] == 'random(1)'
        return {'samples': np.zeros((2,50,3)).tolist()}
    monkeypatch.setattr(MCPGenerator, '_call', call)
    g = MCPGenerator({'video': {'allowed_videos': ['one'], 'dataset_profiles': True}})
    result = g.generate(dict(generator='video', label='walking', count=2, videos=['one'],
                      selection='random(1)', source_dataset='capture_24', channel_names=['ax','ay','az']),50,3)
    assert result.shape == (2,50,3)


def test_service_rejects_bad_worker_output(config, monkeypatch):
    def run(command, **kwargs):
        np.save(command[command.index('--output') + 1], np.full((2,50,3), np.nan))
    monkeypatch.setattr('subprocess.run', run)
    with pytest.raises(ValueError, match='invalid samples'):
        IMUTubeService(config).generate(**arguments())


def test_real_stdio_video_rejects_diagnostic_samples_for_training(config, tmp_path):
    pytest.importorskip('mcp')
    from autohar_researcher.synthetic import MCPGenerator
    worker = tmp_path / 'fixture-python'
    worker.write_text('#!' + sys.executable + '\n' + '''import sys, json
from pathlib import Path
import numpy as np
r = json.loads(Path(sys.argv[sys.argv.index('--request') + 1]).read_text())
assert r['profile']['channel_names'] == ['ax','ay','az']
np.save(sys.argv[sys.argv.index('--output') + 1], np.ones((r['count'],r['window_steps'],r['channels'])))
print('worker log')
''')
    worker.chmod(0o755)
    config['python'] = str(worker)
    path = tmp_path / 'server.json'
    path.write_text(json.dumps(config))
    g = MCPGenerator({'video': {'command': sys.executable,
        'args': ['-m','autohar_researcher.imutube_mcp','--config',str(path)],
        'tool':'generate_imu','allowed_videos':['one'], 'dataset_profiles':True}})
    with pytest.raises(ValueError, match='incomplete or diagnostic-only'):
        g.generate(dict(generator='video', label='walking', count=2, videos=['one'],
                       source_dataset='wisdm', channel_names=['ax','ay','az']),50,3)


def test_external_adapter_subprocess(config, tmp_path):
    adapter = tmp_path / 'adapter.py'
    adapter.write_text(
        "import argparse, json, numpy as np\n"
        "p=argparse.ArgumentParser()\n"
        "for key in ('config','request','output'): p.add_argument('--'+key, required=True)\n"
        "a=p.parse_args()\n"
        "r=json.load(open(a.request))\n"
        "assert r['protocol_version']==1 and r['dataset']=='capture_24'\n"
        "assert r['units']=={'acc':'m/s^2','gyro':'rad/s'}\n"
        "assert r['channel_names']==['ax','ay','az']\n"
        "assert len(r['video_paths'])==1\n"
        "np.save(a.output, np.full((r['count'],r['window_steps'],r['channels']), 2.5))\n"
    )
    config.pop('pose_model')
    config.update(backend='external', external_command=[sys.executable, str(adapter)],
                  implementation='licensed-test-adapter')
    result = IMUTubeService(config).generate(**arguments(), selection='random(1)')
    assert np.asarray(result['samples']).shape == (2, 50, 3)
    assert np.all(np.asarray(result['samples']) == 2.5)
    assert result['implementation'] == 'licensed-test-adapter'
    assert result['coordinate_mode'] == 'unspecified'


@pytest.mark.parametrize('command', [None, '', [], [''], [42]])
def test_external_command_validation(config, command):
    config.update(backend='external', external_command=command)
    with pytest.raises(ValueError, match='external_command'):
        IMUTubeService(config)


@pytest.mark.parametrize('output', [np.zeros((1, 50, 3)), np.full((2, 50, 3), np.nan),
                                    np.zeros((2, 50, 3), dtype=complex)])
def test_external_invalid_output(config, monkeypatch, output):
    config.update(backend='external', external_command=['licensed-adapter'])
    def run(command, **kwargs):
        np.save(command[command.index('--output') + 1], output)
    monkeypatch.setattr('subprocess.run', run)
    with pytest.raises(ValueError, match='invalid samples'):
        IMUTubeService(config).generate(**arguments())


@pytest.mark.parametrize('dataset', list(DATASET_PROFILES))
def test_hip_centering_erases_rigid_body_acceleration_for_every_profile(dataset):
    absolute = poses()
    t = np.arange(len(absolute))/30
    absolute[:, :, 0] += t[:, None]**2
    centered = absolute - ((absolute[:,23]+absolute[:,24])/2)[:,None,:]
    profile = DATASET_PROFILES[dataset]
    world = pose_to_imu(absolute, 30, 30, profile, gravity=[0,0,0], coordinate_mode='absolute-world')
    relative = pose_to_imu(centered, 30, 30, profile, gravity=[0,0,0],
                           diagnostic_only=True, return_provenance=True)
    np.testing.assert_allclose(np.linalg.norm(world[20:-20,:3],axis=1),2,atol=1e-5)
    np.testing.assert_allclose(relative['samples'][20:-20,:3],0,atol=1e-5)
    assert relative['provenance']['common_body_translation_available'] is False
    assert relative['provenance']['training_eligible'] is False
    assert relative['provenance']['physically_complete_imu'] is False
    with pytest.raises(ValueError, match='Hip-relative'):
        pose_to_imu(centered,30,30,profile)


def test_relative_limb_motion_is_preserved_only_as_diagnostic_signal():
    centered = poses()
    t=np.arange(len(centered))/30
    for joint in (14,16,18,20):
        centered[:,joint,0] += .1*t**2
    result=pose_to_imu(centered,30,30,DATASET_PROFILES['utd_mhad'], gravity=[0,0,0],
                       diagnostic_only=True,return_provenance=True)
    np.testing.assert_allclose(result['samples'][20:-20,0],.2,atol=1e-5)
    assert result['provenance']['physically_complete_imu'] is False
    with pytest.raises(ValueError, match='provenance'):
        pose_to_imu(centered,30,30,DATASET_PROFILES['utd_mhad'],diagnostic_only=True)


@pytest.mark.parametrize('dataset', list(DATASET_PROFILES))
def test_service_blocks_incomplete_mediapipe_training_before_launch(config, monkeypatch, dataset):
    config.pop('diagnostic_only')
    # A configuration assertion cannot turn hip-relative MediaPipe output into world motion.
    config.update(coordinate_mode='absolute-world', calibration='claimed calibrated')
    monkeypatch.setattr('subprocess.run', lambda *args,**kwargs:pytest.fail('Unsafe worker was launched'))
    profile=DATASET_PROFILES[dataset]
    args=arguments()
    args.update(dataset=dataset,channels=len(profile['channel_map']),channel_names=profile['channel_names'])
    with pytest.raises(ValueError, match='Hip-relative'):
        IMUTubeService(config).generate(**args)


def test_worker_cannot_bypass_guard_by_claiming_absolute_coordinates(monkeypatch):
    monkeypatch.setattr('autohar_researcher.imutube_worker.extract_segments',
                        lambda *args:pytest.fail('Pose extraction should not start'))
    request=dict(profile=DATASET_PROFILES['uci_har'])
    with pytest.raises(ValueError,match='Hip-relative'):
        generate({'coordinate_mode':'absolute-world','gravity':[0,0,0]},request)


@pytest.mark.parametrize('metadata', [
    {'implementation':'imutube-inspired-mediapipe'},
    {'coordinate_mode':'hip-relative'},
    {'training_eligible':False},
    {'physically_complete_imu':False},
    {'diagnostic_only':True},
    {'provenance':{'training_eligible':False}},
    {'provenance':{'physically_complete_imu':False}},
    {'provenance':{'diagnostic_only':True}},
    {'provenance':{'coordinate_mode':'hip-relative'}},
])
def test_video_client_rejects_incomplete_or_legacy_mediapipe_output(monkeypatch,metadata):
    from autohar_researcher.synthetic import MCPGenerator
    async def call(self,spec,args):
        return {'samples':np.zeros((2,50,3)).tolist(),**metadata}
    monkeypatch.setattr(MCPGenerator,'_call',call)
    generator=MCPGenerator({'video':{'allowed_videos':['one'],'dataset_profiles':True}})
    with pytest.raises(ValueError,match='incomplete or diagnostic-only'):
        generator.generate(dict(generator='video',label='walking',count=2,videos=['one'],
            source_dataset='capture_24',channel_names=['ax','ay','az']),50,3)


def test_mediapipe_waist_diagnostic_provenance_identifies_degenerate_origin():
    result=pose_to_imu(poses(),30,30,DATASET_PROFILES['hapt'],diagnostic_only=True,return_provenance=True)
    assert result['provenance']['waist_origin_degenerate']
    assert not result['provenance']['root_acceleration_recoverable']
    np.testing.assert_allclose(np.linalg.norm(result['samples'][20:-20,:3],axis=1),9.80665,atol=1e-5)


def test_worker_cli_persists_diagnostic_provenance(tmp_path, monkeypatch):
    from autohar_researcher import imutube_worker
    config_path = tmp_path / 'config.json'
    request_path = tmp_path / 'request.json'
    output = tmp_path / 'samples.npy'
    config_path.write_text(json.dumps({'diagnostic_only': True}))
    request_path.write_text(json.dumps({'profile': DATASET_PROFILES['uci_har']}))
    monkeypatch.setattr('sys.argv', ['worker', '--config', str(config_path),
        '--request', str(request_path), '--output', str(output)])
    monkeypatch.setattr(imutube_worker, 'generate', lambda *args: np.zeros((2, 50, 6)))
    imutube_worker.main()
    assert np.load(output).shape == (2, 50, 6)
    provenance = json.loads(output.with_suffix('.provenance.json').read_text())
    assert provenance['coordinate_mode'] == 'hip-relative'
    assert provenance['diagnostic_only'] is True
    assert provenance['training_eligible'] is False
    assert provenance['physically_complete_imu'] is False
    assert provenance['waist_origin_degenerate'] is True
