from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .benchmark_data import csv_recordings, select_subjects, loso_groups


def _whar_recordings(name, spec, raw_dir):
    import whar_datasets as whar
    dataset_id = next(member for member in whar.WHARDatasetID if member.value == name)
    cfg = whar.get_dataset_cfg(dataset_id)
    activity_df, session_df, sessions = cfg.parse(str(raw_dir), cfg.activity_id_col)
    activity_names = dict(zip(activity_df['activity_id'], activity_df['activity_name']))
    rows = session_df.to_dict('records')
    selected = select_subjects([row['subject_id'] for row in rows], spec)
    for row in rows:
        if str(row['subject_id']) not in selected:
            continue
        label = str(activity_names[row['activity_id']])
        if label not in spec['label_map']:
            raise ValueError(f'No predeclared semantic mapping for {name}:{label}')
        if spec['label_map'][label] is None:
            continue
        values = sessions[row['session_id']][spec['source_columns']].to_numpy(dtype=np.float32)
        yield str(row['subject_id']), str(row['session_id']), label, values


def export(spec_path: Path, output: Path) -> Path:
    specs = json.loads(spec_path.read_text())['datasets']
    if output.exists():
        raise ValueError('WHAR export destination must not exist')
    output.mkdir(parents=True)
    datasets, recordings = {}, []
    for name, spec in specs.items():
        raw_dir = (spec_path.parent / spec['raw_dir']).resolve()
        columns = spec['source_columns']
        if len(columns) != len(spec['channels']) or len(set(columns)) != len(columns):
            raise ValueError('Source columns must match the fixed channel layout')
        datasets[name] = {key: spec[key] for key in ('sampling_hz', 'channels', 'unit_scales',
                                                    'placement', 'labels', 'label_map')}
        if 'loso_groups' in spec:
            datasets[name]['loso_groups'] = spec['loso_groups']
        if 'loso_selection_provenance' in spec:
            datasets[name]['loso_selection_provenance'] = spec['loso_selection_provenance']
        adapter = spec.get('adapter', 'whar')
        if adapter not in {'whar', 'hangtime_9', 'capture24_12'}:
            raise ValueError(f'Unknown dataset adapter: {adapter}')
        rows = (_whar_recordings(name, spec, raw_dir) if adapter == 'whar'
                else csv_recordings(raw_dir, spec))
        exported = 0
        for subject, session, label, values in rows:
            mapped = spec['label_map'].get(label)
            if mapped not in spec['labels']:
                raise ValueError(f'Exported label {label!r} is not in the declared label space')
            if not len(values) or not np.isfinite(values).all():
                raise ValueError('Raw session contains empty/non-finite signals')
            relative = Path(name) / f'{session}.npy'
            (output / name).mkdir(exist_ok=True)
            np.save(output / relative, values, allow_pickle=False)
            exported += 1
            recordings.append({'dataset': name, 'subject': subject,
                               'session': session, 'path': str(relative), 'label': label})
        if not exported:
            raise ValueError(f'No recordings retained for {name}')
        if 'loso_groups' in spec:
            loso_groups([r['subject'] for r in recordings if r['dataset'] == name], spec['loso_groups'])
    manifest = output / 'manifest.json'
    manifest.write_text(json.dumps({'datasets': datasets, 'recordings': recordings}, indent=2))
    return manifest


def main():
    parser = argparse.ArgumentParser(description='Export raw recordings with predefined dataset mappings')
    parser.add_argument('--spec', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    print(export(args.spec, args.output))


if __name__ == '__main__':
    main()
