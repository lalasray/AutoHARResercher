import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from autohar_researcher.whar_export import export


def test_whar_export_keeps_raw_sessions_and_explicit_mapping(tmp_path, monkeypatch):
    class Metadata:
        def to_dict(self, orient):
            return [{'session_id': 9, 'subject_id': 3, 'activity_id': 0}]
    class Array:
        def to_numpy(self, dtype):
            return np.ones((50, 3), dtype=dtype)
    class Frame:
        def __getitem__(self, columns):
            assert columns == ['x', 'y', 'z']
            return Array()
    config = SimpleNamespace(activity_id_col='activity_id', parse=lambda *args: (
        {'activity_id': [0], 'activity_name': ['Walking']}, Metadata(), {9: Frame()}))
    fake = SimpleNamespace(WHARDatasetID=[SimpleNamespace(value='uci_har')],
                           get_dataset_cfg=lambda dataset: config)
    monkeypatch.setitem(sys.modules, 'whar_datasets', fake)
    monkeypatch.setattr('autohar_researcher.whar_export.version', lambda name: 'fixture')
    spec = tmp_path / 'spec.json'
    spec.write_text(json.dumps({'datasets': {'uci_har': {
        'raw_dir': 'raw', 'sampling_hz': 50, 'source_columns': ['x', 'y', 'z'],
        'channels': ['ax', 'ay', 'az'], 'unit_scales': [9.80665] * 3,
        'placement': 'waist', 'labels': ['walking'], 'label_map': {'Walking': 'walking'},
    }}}))
    manifest = export(spec, tmp_path / 'prepared')
    value = json.loads(manifest.read_text())
    row = value['recordings'][0]
    assert row['subject'] == '3' and row['session'] == '9' and row['label'] == 'Walking'
    np.testing.assert_array_equal(np.load(manifest.parent / row['path']), np.ones((50, 3)))
    with pytest.raises(ValueError, match='must not exist'):
        export(spec, tmp_path / 'prepared')
