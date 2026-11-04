import json
from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
import pytest

from autohar_researcher.datasets import load_manifest, make_task
from autohar_researcher.imugpt_profiles import DATASET_PROFILES
from autohar_researcher.patches import validate_pipeline
from autohar_researcher.types import LOSO_DATASETS, Pipeline
from autohar_researcher.whar_export import export

EXAMPLES = Path(__file__).resolve().parents[1] / 'examples'
RAW_LABELS = {
    'utd_mhad': ['Swipe Left', 'Swipe Right', 'Wave', 'Clap', 'Throw', 'Arm Cross', 'Basketball Shoot',
                 'Draw X', 'Draw Circle Clockwise', 'Draw Circle Counterclockwise', 'Draw Triangle',
                 'Bowling', 'Boxing', 'Baseball Swing', 'Tennis Swing', 'Arm Curl', 'Tennis Serve',
                 'Two-Hand Push', 'Knock', 'Catch', 'Pick Up and Throw', 'Jog', 'Walk',
                 'Sit to Stand', 'Stand to Sit', 'Lunge', 'Squat'],
    'hapt': 'WALKING WALKING_UPSTAIRS WALKING_DOWNSTAIRS SITTING STANDING LAYING STAND_TO_SIT SIT_TO_STAND SIT_TO_LIE LIE_TO_SIT STAND_TO_LIE LIE_TO_STAND'.split(),
    'usc_had': ['Walking Forward','Walking Left','Walking Right','Walking Upstairs','Walking Downstairs','Running Forward','Jumping Up','Sitting','Standing','Sleeping','Elevator Up','Elevator Down'],
    'unimib_shar': 'StandingUpFS StandingUpFL Walking Running GoingUpS Jumping GoingDownS LyingDownFS SittingDown FallingForw FallingRight FallingBack HittingObstacle FallingWithPS FallingBackSC Syncope FallingLeft'.split(),
}


@pytest.mark.parametrize('name', RAW_LABELS)
def test_profile_export_and_loso(name, tmp_path, monkeypatch):
    pd = pytest.importorskip('pandas')
    path = EXAMPLES / ('whar-' + name.replace('_','-') + '.json')
    spec = json.loads(path.read_text())['datasets'][name]
    columns = {'hapt': ['acc_x','acc_y','acc_z','gyro_x','gyro_y','gyro_z'],
               'utd_mhad': ['Ax','Ay','Az','GyroX','GyroY','GyroZ'],
               'usc_had': ['Ax','Ay','Az','GyroX','GyroY','GyroZ'],
               'unimib_shar': ['acc_x','acc_y','acc_z']}[name]
    labels = RAW_LABELS[name]
    metadata, sessions = [], {}
    selected_subjects = len(spec['subjects']) if 'subjects' in spec else spec.get('first_n_subjects', 3)
    if 'loso_groups' in spec:
        selected_subjects = sum(map(len, spec['loso_groups']))
    available_subjects = selected_subjects + 1 if 'first_n_subjects' in spec else selected_subjects
    for subject in range(available_subjects):
        for activity in range(len(labels)):
            session = len(metadata)
            metadata.append(dict(subject_id=subject, session_id=session, activity_id=activity))
            sessions[session] = pd.DataFrame(np.ones((int(spec['sampling_hz'] * 3),len(columns))), columns=columns)
    cfg = SimpleNamespace(activity_id_col='activity_id', parse=lambda *args: (
        pd.DataFrame({'activity_id': range(len(labels)), 'activity_name': labels}),
        pd.DataFrame(metadata), sessions))
    monkeypatch.setitem(sys.modules,'whar_datasets',SimpleNamespace(
        WHARDatasetID=[SimpleNamespace(value=name)], get_dataset_cfg=lambda _:cfg))
    manifest = export(path,tmp_path/'prepared')
    exported, records = load_manifest(manifest)
    if 'loso_groups' in spec:
        assert exported['datasets'][name]['loso_groups'] == spec['loso_groups']
    retained_labels = [label for label in labels if spec['label_map'][label] is not None]
    assert len(records) == len(retained_labels)*selected_subjects
    assert {r.label for r in records} == set(spec['labels'])
    assert {r.subject for r in records} == {str(i) for i in range(selected_subjects)}
    signal = records[0].load()
    assert signal.shape == (150,len(columns))
    expected = {'hapt':[9.80665]*3+[1]*3,
                'utd_mhad':[9.80665]*3+[np.pi/180]*3,
                'usc_had':[9.80665]*3+[np.pi/180]*3,
                'unimib_shar':[1]*3}[name]
    np.testing.assert_allclose(signal[30:-30], np.tile(expected,(90,1)),rtol=.002)
    task = make_task(manifest,'loso',name,'0')
    assert set(task.labels) == set(spec['labels'])
    if name == 'unimib_shar':
        assert len(task.labels) == 9
        assert list(task.labels) == [spec['label_map'][label] for label in labels[:9]]
        assert all(spec['label_map'][label] is None for label in labels[9:])
    if name == 'utd_mhad':
        assert len(task.labels) == 11
        assert all(spec['label_map'][label] == 'drawing' for label in labels[7:11])
        assert {spec['label_map'][label] for label in ('Jog', 'Walk', 'Squat')} == {'jogging', 'walking', 'squat'}
        assert all(spec['label_map'][label] is None for label in ('Sit to Stand', 'Stand to Sit', 'Lunge'))
        assert spec['placement'] == 'right wrist'
    assert {r.subject for r in task.test} == set(spec.get('loso_groups', [['0']])[0])
    assert not ({r.subject for r in task.train} & {r.subject for r in task.validation})


def test_all_ten_profiles_have_complete_matching_synthetic_policies():
    union = set()
    for name in LOSO_DATASETS:
        spec = json.loads((EXAMPLES / ('whar-'+name.replace('_','-')+'.json')).read_text())['datasets'][name]
        policy = Pipeline.from_dict(json.loads((EXAMPLES/'synthetic-policies'/f'{name}.json').read_text()))
        validate_pipeline(policy,{name})
        requests = policy.augmentation['synthetic']
        assert len(requests) == len(spec['labels'])
        assert {r['label'] for r in requests} == set(spec['labels'])
        assert spec['channels'] == DATASET_PROFILES[name]['channel_names']
        assert all(25 <= r['count'] <= 200 and .05 <= r['ratio'] <= .5 and r['prompt'].strip() for r in requests)
        union.update(spec['labels'])
    catalog = json.loads((EXAMPLES/'synthetic-policy.json').read_text())['augmentation']['synthetic']
    assert len(catalog) == len({r['label'] for r in catalog})
    assert union == {r['label'] for r in catalog}
