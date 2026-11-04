import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from autohar_researcher.whar_export import export


def test_groups_resolve_individual_defaults_and_partial_selection():
    from autohar_researcher.benchmark_data import loso_groups
    people = ['10', '2', '1', '20', '3']
    assert loso_groups(people) == [['1'], ['2'], ['3'], ['10'], ['20']]
    assert loso_groups(people, [['20', '2'], ['3']]) == [['20', '2'], ['3']]


@pytest.mark.parametrize('excluded', [False, True])
def test_whar_export_keeps_raw_sessions_and_explicit_mapping(tmp_path, monkeypatch, excluded):
    class Metadata:
        def to_dict(self, orient):
            return [{'session_id': 9, 'subject_id': 3, 'activity_id': 0}] + (
                [{'session_id': 10, 'subject_id': 3, 'activity_id': 1},
                 {'session_id': 11, 'subject_id': 4, 'activity_id': 0}] if excluded else [])
    class Array:
        def to_numpy(self, dtype):
            return np.ones((50, 3), dtype=dtype)
    class Frame:
        def __getitem__(self, columns):
            assert columns == ['x', 'y', 'z']
            return Array()
    config = SimpleNamespace(activity_id_col='activity_id', parse=lambda *args: (
        {'activity_id': [0, 1], 'activity_name': ['Walking', 'Inactive']}, Metadata(), {9: Frame()}))
    fake = SimpleNamespace(WHARDatasetID=[SimpleNamespace(value='uci_har')],
                           get_dataset_cfg=lambda dataset: config)
    monkeypatch.setitem(sys.modules, 'whar_datasets', fake)
    spec = tmp_path / 'spec.json'
    spec.write_text(json.dumps({'datasets': {'uci_har': {
        'raw_dir': 'raw', 'sampling_hz': 50, 'source_columns': ['x', 'y', 'z'],
        'channels': ['ax', 'ay', 'az'], 'unit_scales': [9.80665] * 3,
        'placement': 'waist', 'labels': ['walking'], 'label_map': {'Walking': 'walking', 'Inactive': None}, 'subjects': ['3'],
        'loso_groups': [['3']],
    }}}))
    manifest = export(spec, tmp_path / 'prepared')
    value = json.loads(manifest.read_text())
    assert value['datasets']['uci_har']['loso_groups'] == [['3']]
    assert len(value['recordings']) == 1
    row = value['recordings'][0]
    assert row['subject'] == '3' and row['session'] == '9' and row['label'] == 'Walking'
    np.testing.assert_array_equal(np.load(manifest.parent / row['path']), np.ones((50, 3)))
    with pytest.raises(ValueError, match='must not exist'):
        export(spec, tmp_path / 'prepared')


def test_real_life_config_exports_three_class_mapping(tmp_path, monkeypatch):
    from pathlib import Path
    spec = Path(__file__).resolve().parents[1] / 'examples/whar-real-life-har.json'
    class Metadata:
        def to_dict(self, orient):
            return [{'session_id': person * 4 + i, 'subject_id': person, 'activity_id': i}
                    for person in range(17) for i in range(4)]
    class Frame:
        def __getitem__(self, columns):
            assert columns == ['acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z']
            return self
        def to_numpy(self, dtype):
            return np.ones((40, 6), dtype=dtype)
    config = SimpleNamespace(activity_id_col='activity_id', parse=lambda *args: (
        {'activity_id': list(range(4)), 'activity_name': ['Inactive', 'Active', 'Walking', 'Driving']},
        Metadata(), {i: Frame() for i in range(68)}))
    fake = SimpleNamespace(WHARDatasetID=[SimpleNamespace(value='real_life_har')],
                           get_dataset_cfg=lambda dataset: config)
    monkeypatch.setitem(sys.modules, 'whar_datasets', fake)
    manifest = export(spec, tmp_path / 'prepared')
    from autohar_researcher.datasets import load_manifest
    raw, records = load_manifest(manifest)
    assert [r.label for r in records] == ['standing', 'walking', 'sitting'] * 17
    assert [len(g) for g in raw['datasets']['real_life_har']['loso_groups']] == [4, 4, 4, 5]
    assert all(r.hz == 20 for r in records)
    assert raw['datasets']['real_life_har']['labels'] == ['sitting', 'standing', 'walking']
    assert not (manifest.parent / 'real_life_har/0.npy').exists()


def test_subject_selection_is_natural_and_strict():
    from autohar_researcher.benchmark_data import select_subjects
    assert select_subjects(['10', '2', '1', '20', '3'], {'first_n_subjects': 4}) == ['1', '2', '3', '10']
    with pytest.raises(ValueError, match='available'):
        select_subjects(['0', '1'], {'subjects': ['0', '2']})
    with pytest.raises(ValueError, match='first_n_subjects'):
        select_subjects(['0'], {'first_n_subjects': 4})


def test_hangtime_combines_tiers_and_preserves_exclusion_boundaries(tmp_path, monkeypatch):
    pd = pytest.importorskip('pandas')
    from autohar_researcher.benchmark_data import HANGTIME_LABELS, csv_recordings
    frame = pd.DataFrame({'timestamp': np.arange(8) * 20,
        'acc_x': np.arange(8), 'acc_y': np.ones(8), 'acc_z': np.ones(8),
        'basketball': ['not_labeled', 'shot', 'not_labeled', 'not_labeled',
                       'not_labeled', 'pass', 'pass', 'pass'],
        'locomotion': ['walking', 'jumping', 'jumping', 'walking', 'walking', 'standing', 'standing', 'standing'],
        'coarse': ['warmup'] * 7 + ['game']})
    frame.to_csv(tmp_path / 'player1.csv', index=False)
    spec = {'adapter': 'hangtime_9', 'sampling_hz': 50, 'timestamp_unit': 'ms',
            'source_columns': ['acc_x', 'acc_y', 'acc_z']}
    # Force several chunk boundaries, including inside a continuous activity.
    read_csv = pd.read_csv
    monkeypatch.setattr(pd, 'read_csv', lambda *a, **kw: read_csv(*a, **(kw | {'chunksize': 2})))
    rows = list(csv_recordings(tmp_path, spec))
    assert [r[2] for r in rows] == ['walking', 'shot', 'walking', 'pass', 'pass']
    assert [len(r[3]) for r in rows] == [1, 1, 2, 2, 1]
    assert len(HANGTIME_LABELS) == 9


def test_capture24_groups():
    from autohar_researcher.benchmark_data import capture24_group, CAPTURE24_LABELS
    examples = ['sleeping', 'sitting reading', 'standing talking', 'walking the dog',
                'bicycling', 'sitting in a car', 'standing washing dishes', 'running',
                'sitting at computer', 'sitting watching TV', 'walking upstairs', 'personal care']
    assert tuple(map(capture24_group, examples)) == CAPTURE24_LABELS
    assert capture24_group('unknown') is None
    with pytest.raises(ValueError, match='dictionary'):
        capture24_group('12345')


def test_capture24_exports_all_subjects_with_four_loso_groups(tmp_path):
    pd = pytest.importorskip('pandas')
    from pathlib import Path
    from autohar_researcher.datasets import load_manifest
    spec = json.loads((Path(__file__).resolve().parents[1] / 'examples/whar-capture-24.json').read_text())
    spec['datasets']['capture_24']['raw_dir'] = 'raw'
    raw = tmp_path / 'raw'
    raw.mkdir()
    selected = [person for group in spec['datasets']['capture_24']['loso_groups'] for person in group]
    for subject in selected + ['P151']:
        pd.DataFrame({'time': pd.to_datetime([0, 10, 20, 30], unit='ms'),
            'x': [1.] * 4, 'y': [2.] * 4, 'z': [3.] * 4,
            'annotation': ['walking', 'unknown', 'walking', 'sitting at computer']}).to_csv(raw / f'{subject}.csv', index=False)
    path = tmp_path / 'spec.json'
    path.write_text(json.dumps(spec))
    manifest = export(path, tmp_path / 'prepared')
    value, rows = load_manifest(manifest)
    assert {r.subject for r in rows} == set(selected + ['P151'])
    groups = value['datasets']['capture_24']['loso_groups']
    assert [len(group) for group in groups] == [5]*4
    from autohar_researcher.datasets import make_task
    for group in groups:
        task = make_task(manifest, 'loso', 'capture_24', group[0])
        assert {r.subject for r in task.test} == set(group)
        assert {r.subject for r in task.train + task.validation} == set(selected + ['P151']) - set(group)
    assert len(value['datasets']['capture_24']['labels']) == 12
    assert len(rows) == 63  # Unknown interval separates the two walking recordings.
    assert 'provenance' not in value
    assert {r.label for r in rows} == {'walking', 'computer_use'}
    assert all(r.load().shape[1] == 3 for r in rows)


def test_missing_requested_utd_subject_is_rejected():
    from pathlib import Path
    from autohar_researcher.benchmark_data import select_subjects
    config = json.loads((Path(__file__).resolve().parents[1] / 'examples/whar-utd-mhad.json').read_text())['datasets']['utd_mhad']
    assert select_subjects(range(8), config) == list(map(str, range(8)))
    with pytest.raises(ValueError, match='available subject IDs'):
        select_subjects(range(7), config)
