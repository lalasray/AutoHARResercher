"""Export raw WHAR parser sessions with explicit, predeclared harmonization.

This preparation command does not use WHAR's window splitter or normalization.
Raw datasets must already be downloaded and extracted at the parser's expected
path. Dataset licenses and acquisition are managed outside the search process.
"""
from __future__ import annotations

import argparse
import json
from importlib.metadata import version
from pathlib import Path

import numpy as np


def export(spec_path: Path, output: Path) -> Path:
    import whar_datasets as whar
    specs = json.loads(spec_path.read_text())['datasets']
    if output.exists():
        raise ValueError('WHAR export destination must not exist')
    output.mkdir(parents=True)
    datasets, recordings = {}, []
    for name, spec in specs.items():
        dataset_id = next(member for member in whar.WHARDatasetID if member.value == name)
        cfg = whar.get_dataset_cfg(dataset_id)
        raw_dir = (spec_path.parent / spec['raw_dir']).resolve()
        activity_df, session_df, sessions = cfg.parse(str(raw_dir), cfg.activity_id_col)
        activity_names = dict(zip(activity_df['activity_id'], activity_df['activity_name']))
        columns = spec['source_columns']
        if len(columns) != len(spec['channels']) or len(set(columns)) != len(columns):
            raise ValueError('Source columns must match the fixed channel layout')
        # Never guess sensor units, label correspondences, or placement.
        datasets[name] = {key: spec[key] for key in ('sampling_hz', 'channels', 'unit_scales',
                                                    'placement', 'labels', 'label_map')}
        for row in session_df.to_dict('records'):
            session_id = row['session_id']
            label = str(activity_names[row['activity_id']])
            if label not in spec['label_map']:
                raise ValueError(f'No predeclared semantic mapping for {name}:{label}')
            frame = sessions[session_id]
            values = frame[columns].to_numpy(dtype=np.float32)
            if not len(values) or not np.isfinite(values).all():
                raise ValueError('Raw WHAR session contains empty/non-finite signals')
            relative = Path(name) / f'{session_id}.npy'
            (output / name).mkdir(exist_ok=True)
            np.save(output / relative, values, allow_pickle=False)
            recordings.append({'dataset': name, 'subject': str(row['subject_id']),
                               'session': str(session_id), 'path': str(relative), 'label': label})
    manifest = output / 'manifest.json'
    manifest.write_text(json.dumps({'datasets': datasets, 'recordings': recordings,
        'provenance': {'adapter': 'whar raw parser sessions', 'whar_version': version('whar-datasets'),
                       'harmonization_spec': specs}}, indent=2))
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spec', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    print(export(args.spec, args.output))


if __name__ == '__main__':
    main()
