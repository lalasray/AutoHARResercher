import json

import numpy as np
import pytest


@pytest.fixture
def manifest(tmp_path):
    datasets = {}
    rows = []
    for dataset in ('uci_har', 'motion_sense', 'real_life_har'):
        labels = ['walking', 'walking_upstairs', 'walking_downstairs', 'sitting', 'standing']
        datasets[dataset] = {'sampling_hz': 50, 'channels': ['ax', 'ay', 'az', 'gx', 'gy', 'gz'],
            'unit_scales': [1] * 6, 'placement': 'waist', 'labels': labels,
            'label_map': {str(i): name for i, name in enumerate(labels)}}
        for subject in range(6):
            for label in range(5):
                path = tmp_path / f'{dataset}-{subject}-{label}.npy'
                rng = np.random.default_rng(subject * 5 + label)
                np.save(path, rng.normal(label, .1, (100, 6)).astype(np.float32))
                rows.append({'dataset': dataset, 'subject': str(subject),
                    'session': f'{subject}-{label}', 'label': str(label), 'path': path.name})
    path = tmp_path / 'manifest.json'
    path.write_text(json.dumps({'datasets': datasets, 'recordings': rows}))
    return path
