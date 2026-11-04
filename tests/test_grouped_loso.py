import json
from pathlib import Path

import pytest

from autohar_researcher.benchmark_data import loso_groups
from autohar_researcher.datasets import make_task, make_fixed_validation_task
from autohar_researcher.loso import LOSOHarness


def test_group_split_and_harness(manifest, tmp_path):
    raw = json.loads(manifest.read_text())
    groups = [['0', '1'], ['2', '3'], ['4', '5']]
    raw['datasets']['uci_har']['loso_groups'] = groups
    manifest.write_text(json.dumps(raw))
    tasks = [make_task(manifest, 'loso', 'uci_har', g[0]) for g in groups]
    assert make_task(manifest, 'loso', 'uci_har', '1').fingerprint == tasks[0].fingerprint
    for group, task in zip(groups, tasks):
        assert {r.subject for r in task.test} == set(group)
        assert not set(group) & {r.subject for r in task.train + task.validation}
        assert not {r.subject for r in task.train} & {r.subject for r in task.validation}
    harness = LOSOHarness(tasks, 13, tmp_path / 'folds', 'cpu')
    assert len(harness.folds) == 3
    assert harness.available_subject_count == 6
    fixed = make_fixed_validation_task(manifest, 'uci_har', '0')
    assert {r.subject for r in fixed.validation} == {'0'}


@pytest.mark.parametrize('groups', [[], [['0'], ['0', '1']], [['0', '1', '2']], ['0']])
def test_invalid_groups(groups):
    with pytest.raises(ValueError):
        loso_groups(['0', '1'], groups)


def test_configured_default_groups():
    root = Path(__file__).resolve().parents[1] / 'examples'
    sizes = {'wisdm': [4]*6+[3]*4, 'uci_har': [3]*10, 'motion_sense': [4]*6,
             'hapt': [3]*10, 'usc_had': [2]*7}
    for name, expected in sizes.items():
        spec = json.loads((root / ('whar-'+name.replace('_', '-')+'.json')).read_text())['datasets'][name]
        groups = loso_groups(map(str, range(sum(expected))), spec['loso_groups'])
        assert list(map(len, groups)) == expected
        assert [p for g in groups for p in g] == list(map(str, range(sum(expected))))


def test_cli_deduplicates_group_folds(manifest, monkeypatch, capsys):
    import sys
    from autohar_researcher.main import main
    raw = json.loads(manifest.read_text())
    raw['datasets']['uci_har']['loso_groups'] = [['0','1'], ['2','3'], ['4','5']]
    manifest.write_text(json.dumps(raw))
    monkeypatch.setattr(sys, 'argv', ['autohar', 'splits', '--manifest', str(manifest),
                                    '--protocol', 'loso', '--target', 'uci_har'])
    main()
    folds = json.loads(capsys.readouterr().out)
    assert len(folds) == 3
    assert [{row[1] for row in fold['test']} for fold in folds] == [{'0','1'}, {'2','3'}, {'4','5'}]


def test_grouped_subset_keeps_other_training_subjects(manifest, tmp_path):
    raw = json.loads(manifest.read_text())
    spec = raw['datasets']['uci_har']
    spec['loso_groups'] = [['0', '1'], ['2', '3']]
    manifest.write_text(json.dumps(raw))
    tasks = [make_task(manifest, 'loso', 'uci_har', person) for person in ('0', '2')]
    for task in tasks:
        assert {'4', '5'} <= {r.subject for r in task.train + task.validation}
        assert not {r.subject for r in task.test} & {r.subject for r in task.train + task.validation}
    h = LOSOHarness(tasks, 13, tmp_path / 'partial', 'cpu', allow_partial_folds=True)
    assert h.available_subject_count == 6


def test_new_four_group_defaults():
    import numpy as np
    root = Path(__file__).resolve().parents[1] / 'examples'
    for name, sizes in [('real_life_har', [4,4,4,5]), ('hang_time', [7,5,6,6]), ('capture_24', [5]*4)]:
        spec = json.loads((root / ('whar-'+name.replace('_','-')+'.json')).read_text())['datasets'][name]
        groups = spec['loso_groups']
        assert list(map(len, groups)) == sizes
        flat = [person for group in groups for person in group]
        assert len(flat) == len(set(flat))
        if name == 'hang_time':
            assigned = {}
            for index, group in enumerate(groups):
                for person in group:
                    prefix = person.rsplit('_', 1)[0]
                    assert assigned.setdefault(prefix, index) == index
        if name == 'capture_24':
            expected = sorted(np.random.RandomState(42).choice([f'P{i:03d}' for i in range(1,101)],20,replace=False).tolist())
            assert flat == expected


def test_grouped_final_test_aggregates_folds(manifest, tmp_path, monkeypatch):
    import sys
    from autohar_researcher.main import main
    raw = json.loads(manifest.read_text())
    raw['datasets']['uci_har']['loso_groups'] = [['0','1'], ['2','3'], ['4','5']]
    manifest.write_text(json.dumps(raw))
    baseline = tmp_path / 'baseline.json'
    baseline.write_text(json.dumps({'data': {'window_seconds': 1},
        'model': {'architecture': 'cnn', 'hidden_dim': 4, 'depth': 1, 'max_epochs': 1}}))
    output = tmp_path / 'run'
    monkeypatch.setattr(sys, 'argv', ['autohar', 'search', '--manifest', str(manifest),
        '--protocol', 'loso', '--target', 'uci_har', '--method', 'fixed',
        '--baseline', str(baseline), '--device', 'cpu', '--seeds', '13',
        '--output', str(output), '--final-test'])
    main()
    report = json.loads((output / 'summary.json').read_text())['runs'][0]
    assert report['fold_count'] == 3
    assert report['tested_subjects'] == list('012345')
    assert [r['test_subjects'] for r in report['final_loso']] == [['0','1'], ['2','3'], ['4','5']]
    assert report['test_macro_f1'] == pytest.approx(sum(r['test_macro_f1'] for r in report['final_loso']) / 3)
    root = output / 'seed-13' / 'uci_har'
    selected = json.loads((root / 'selected.json').read_text())
    evidence = selected['validation_preprocessing']
    assert evidence['fit_scope'] == 'source_training_only_per_fold'
    assert len(evidence['folds']) == 3
    assert all(ref['available'] and (root / ref['path']).is_file() for ref in evidence['folds'])
    assert [ref['test_subjects'] for ref in evidence['folds']] == [['0','1'], ['2','3'], ['4','5']]
    identity = report['data_identity']
    assert identity['fingerprint_scheme'] == 'recording-content-and-split-v1'
    assert identity['data_fingerprint'] and identity['split_fingerprint'] and identity['manifest_sha256']
    assert selected['data_identity'] == identity
    run = json.loads((root / 'run.json').read_text())
    assert run['data_identity'] == identity
    ledger = json.loads((root / run['data_provenance_file']).read_text())
    assert ledger['data_fingerprint'] == identity['data_fingerprint']
    assert ledger['manifest_sha256'] == identity['manifest_sha256']
    assert len(ledger['recordings']) == len(json.loads(manifest.read_text())['recordings'])
    assert all(row['sha256'] and row['size_bytes'] > 0 for row in ledger['recordings'])
    assert all(row['data_identity']['data_fingerprint'] == identity['data_fingerprint'] for row in report['final_loso'])
