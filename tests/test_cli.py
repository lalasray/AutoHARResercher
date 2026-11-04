import json
import os
import pytest
import subprocess
import sys


def test_loso_subset_cli_option_removed(manifest, monkeypatch, capsys):
    from autohar_researcher.main import main
    monkeypatch.setattr(sys, 'argv', ['autohar', 'splits', '--manifest', str(manifest),
        '--protocol', 'loso', '--loso-subset', '0', '1'])
    with pytest.raises(SystemExit) as error:
        main()
    assert error.value.code == 2
    assert 'unrecognized arguments: --loso-subset' in capsys.readouterr().err


@pytest.mark.parametrize('mode', ['loso', 'fixed-validation'])
def test_explicit_loso_cohort_keeps_all_training_sources(manifest, tmp_path, mode):
    people = ['0', '3', '4', '5']
    data = json.loads(manifest.read_text())
    data['datasets']['uci_har']['loso_groups'] = [[person] for person in people]
    manifest.write_text(json.dumps(data))
    baseline = tmp_path / 'small.json'
    baseline.write_text(json.dumps({'data': {'window_seconds': 1},
        'model': {'architecture': 'cnn', 'hidden_dim': 4, 'depth': 1, 'max_epochs': 1}}))
    output = tmp_path / 'cohort'
    result = subprocess.run([sys.executable, '-m', 'autohar_researcher.main', 'search',
        '--manifest', str(manifest), '--protocol', 'loso', '--target', 'uci_har',
        '--search-mode', mode,
        '--method', 'fixed', '--baseline', str(baseline), '--seeds', '13',
        '--device', 'cpu', '--output', str(output), '--final-test'],
        capture_output=True, text=True, timeout=90)
    assert result.returncode == 0, result.stderr
    root = output / 'seed-13' / ('uci_har' if mode == 'loso' else 'fixed-validation')
    splits = json.loads((root / 'splits.json').read_text())
    if mode == 'loso':
        assert {row[1] for fold in splits for row in fold['test']} == set(people)
        assert len(splits) == 4
        for fold in splits:
            test = {row[1] for row in fold['test']}
            assert {row[1] for row in fold['train'] + fold['validation']} == set('012345') - test
    else:
        assert {row[1] for row in splits['train'] + splits['validation']} == set('012345')
    metadata = json.loads((root / 'run.json').read_text())
    assert metadata['test_subjects'] == people
    assert metadata['available_subject_count'] == 6
    report = json.loads((output / 'summary.json').read_text())['runs'][0]
    assert [fold['subject'] for fold in report['final_loso']] == people


@pytest.mark.parametrize('people', [['0', '0'], ['99'], []])
def test_invalid_loso_cohort_rejected(manifest, monkeypatch, people):
    from autohar_researcher import main
    data = json.loads(manifest.read_text())
    data['datasets']['uci_har']['loso_groups'] = [[person] for person in people]
    manifest.write_text(json.dumps(data))
    monkeypatch.setattr(sys, 'argv', ['autohar', 'splits', '--manifest', str(manifest),
                                     '--protocol', 'loso', '--target', 'uci_har'])
    with pytest.raises(SystemExit) as error:
        main.main()
    assert error.value.code == 2


def test_configured_groups_retain_non_test_sources(manifest, tmp_path):
    data = json.loads(manifest.read_text())
    data['datasets']['uci_har']['loso_groups'] = [['0'], ['1']]
    manifest.write_text(json.dumps(data))
    baseline = tmp_path / 'small.json'
    baseline.write_text(json.dumps({'data': {'window_seconds': 1},
        'model': {'architecture': 'cnn', 'hidden_dim': 4, 'depth': 1, 'max_epochs': 1}}))
    output = tmp_path / 'limited-loso'
    result = subprocess.run([sys.executable, '-m', 'autohar_researcher.main', 'search',
        '--manifest', str(manifest), '--protocol', 'loso', '--target', 'uci_har',
        '--method', 'fixed', '--baseline', str(baseline), '--seeds', '13',
        '--device', 'cpu', '--output', str(output), '--final-test'],
        capture_output=True, text=True, timeout=90)
    assert result.returncode == 0, result.stderr
    root = output / 'seed-13' / 'uci_har'
    metadata = json.loads((root / 'run.json').read_text())
    assert metadata['test_subjects'] == ['0', '1']
    assert metadata['available_subject_count'] == 6
    splits = json.loads((root / 'splits.json').read_text())
    assert len(splits) == 2
    for split in splits:
        test = {r[1] for r in split['test']}
        assert test <= {'0', '1'}
        assert {r[1] for r in split['train'] + split['validation']} == set('012345') - test
    report = json.loads((root / 'final-test.json').read_text())
    assert report['tested_subjects'] == ['0', '1']
    assert report['available_subject_count'] == 6
    assert len(report['final_loso']) == 2


def test_real_cli_search_and_final_test(manifest, tmp_path):
    backend = tmp_path / 'codex-fixture'
    backend.write_text('#!' + sys.executable + '\n' + r'''import json, sys
from pathlib import Path
prompt = sys.stdin.read()
if 'Plan this round.' in prompt:
    response = {'priority': ['data', 'model', 'augmentation']}
elif 'specialist. Own only' in prompt:
    role = prompt.split(' specialist. Own only ')[1].split()[1]
    changes = {'data': {'overlap': .25}, 'model': {'hidden_dim': 8}, 'augmentation': {'kind': 'jitter'}}[role]
    response = {'changes': changes, 'files': {}, 'rationale': 'fixture'}
elif 'Independently judge' in prompt:
    response = {'novelty': 3, 'plausibility': 3, 'compatibility': 4, 'rationale': 'fixture'}
else:
    response = {'subsets': [['data', 'model']]}
Path(sys.argv[sys.argv.index('--output-last-message') + 1]).write_text(json.dumps(response))
''')
    backend.chmod(0o755)
    if os.name == 'nt':
        script = backend.with_suffix('.py')
        backend.rename(script)
        backend = backend.with_suffix('.cmd')
        backend.write_text('@"' + sys.executable + '" "' + str(script) + '" %*\n')
    baseline = tmp_path / 'baseline.json'
    baseline.write_text(json.dumps({'model': {'architecture': 'cnn', 'hidden_dim': 4,
        'depth': 1, 'max_epochs': 1}, 'data': {'window_seconds': 1}}))
    output = tmp_path / 'cli'
    result = subprocess.run([sys.executable, '-m', 'autohar_researcher.main', 'search',
            '--manifest', str(manifest), '--protocol', 'loso', '--target', 'uci_har',
            '--subject', '0', '--seeds', '13', '--rounds', '1', '--budget', '2',
            '--proposal-mode', 'configuration', '--device', 'cpu', '--baseline', str(baseline), '--output', str(output), '--final-test',
            '--backend', 'codex', '--codex-bin', str(backend), '--model', 'test'],
            capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr
    summary = json.loads((output / 'summary.json').read_text())
    assert summary['runs'][0]['evaluations'] == 2
    assert len(summary['runs']) == 1
    assert summary['runs'][0]['fold_count'] == 6
    assert [r['subject'] for r in summary['runs'][0]['final_loso']] == ['0']
    assert 0 <= summary['test_mean'] <= 1
    history = (output / 'seed-13/uci_har/history.jsonl').read_text()
    assert 'test_macro_f1' not in history


def test_loso_cli_defaults_to_independent_folds(manifest):
    result = subprocess.run([sys.executable, '-m', 'autohar_researcher.main', 'splits',
        '--manifest', str(manifest), '--protocol', 'loso', '--target', 'uci_har'],
        capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    folds = json.loads(result.stdout)
    assert len(folds) == 6
    test_people = [next(iter({row[1] for row in fold['test']})) for fold in folds]
    val_people = [next(iter({row[1] for row in fold['validation']})) for fold in folds]
    assert sorted(test_people) == list('012345')
    assert all(test != val for test, val in zip(test_people, val_people))


def test_fixed_validation_search_and_final_loso(manifest, tmp_path):
    baseline = tmp_path / 'small.json'
    baseline.write_text(json.dumps({'model': {'architecture': 'cnn', 'hidden_dim': 4,
        'depth': 1, 'max_epochs': 1}, 'data': {'window_seconds': 1}}))
    output = tmp_path / 'fixed-validation'
    result = subprocess.run([sys.executable, '-m', 'autohar_researcher.main', 'search',
        '--manifest', str(manifest), '--protocol', 'loso', '--search-mode', 'fixed-validation', '--validation-subject', '0',
        '--method', 'fixed', '--baseline', str(baseline), '--seeds', '13',
        '--device', 'cpu', '--output', str(output), '--final-test'],
        capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr
    root = output / 'seed-13/fixed-validation'
    split = json.loads((root / 'splits.json').read_text())
    assert {r[1] for r in split['validation']} == {'0'}
    assert {r[1] for r in split['train']} == set('12345')
    assert split['test'] == []
    summary = json.loads((output / 'summary.json').read_text())
    assert len(summary['runs']) == 1
    assert len(summary['runs'][0]['final_loso']) == 6
    for person in '012345':
        fold = json.loads((root / f'final-loso/subject-{person}/splits.json').read_text())
        assert {r[1] for r in fold['train'] + fold['validation']} == set('012345') - {person}
        assert {r[1] for r in fold['test']} == {person}


def test_fixed_validation_default_selection_is_stable(manifest):
    from autohar_researcher.datasets import make_fixed_validation_task
    from autohar_researcher.types import SEEDS
    assert SEEDS == (13, 14, 15, 16, 17)
    task = make_fixed_validation_task(manifest, 'uci_har')
    assert task == make_fixed_validation_task(manifest, 'uci_har')
    assert len({r.subject for r in task.validation}) == 1
    assert len({r.subject for r in task.train}) == 5
    assert not task.test


@pytest.mark.parametrize('subject', [None, '0'])
def test_independent_loso_keeps_separate_searches(manifest, tmp_path, subject):
    baseline = tmp_path / 'baseline.json'
    baseline.write_text(json.dumps({'model': {'architecture': 'cnn', 'hidden_dim': 4,
        'depth': 1, 'max_epochs': 1}, 'data': {'window_seconds': 1}}))
    output = tmp_path / 'independent'
    result = subprocess.run([sys.executable, '-m', 'autohar_researcher.main', 'search',
        '--manifest', str(manifest), '--protocol', 'loso', '--search-mode', 'independent-loso',
        '--method', 'fixed', '--baseline', str(baseline), '--seeds', '13',
        '--device', 'cpu', '--output', str(output), '--final-test',
        *(['--all-subjects'] if subject is None else ['--subject', subject])],
        capture_output=True, text=True, timeout=90)
    assert result.returncode == 0, result.stderr
    summary = json.loads((output / 'summary.json').read_text())
    expected = set('012345') if subject is None else {subject}
    assert {run['subject'] for run in summary['runs']} == expected
    assert all(run['fold_count'] == 1 for run in summary['runs'])
    for subject in expected:
        root = output / 'seed-13' / f'subject-{subject}'
        assert (root / 'selected.json').exists()
        assert (root / 'final-test.json').exists()


def test_split_preview_matches_explicit_fixed_validation(manifest):
    from autohar_researcher.datasets import make_fixed_validation_task
    result = subprocess.run([sys.executable, '-m', 'autohar_researcher.main', 'splits',
        '--manifest', str(manifest), '--protocol', 'loso', '--search-mode', 'fixed-validation', '--validation-subject', '0'],
        capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    expected = make_fixed_validation_task(manifest, 'uci_har', '0').split_manifest()
    assert json.loads(result.stdout) == json.loads(json.dumps([expected]))


def test_lodo_all_targets_search_and_final_test(manifest, tmp_path):
    import numpy as np
    value = json.loads(manifest.read_text())
    value['datasets']['real_life_har']['labels'] = ['walking', 'sitting', 'standing']
    value['recordings'] = [r for r in value['recordings']
        if r['dataset'] != 'real_life_har' or r['label'] in {'0', '3', '4'}]
    manifest.write_text(json.dumps(value))
    for row in value['recordings']:
        path = manifest.parent / row['path']
        np.save(path, np.tile(np.load(path), (7, 1)))
    baseline = tmp_path / 'small-lodo.json'
    baseline.write_text(json.dumps({'model': {'architecture': 'cnn', 'hidden_dim': 4,
        'depth': 1, 'max_epochs': 1}, 'data': {'window_seconds': 1}}))
    output = tmp_path / 'lodo'
    result = subprocess.run([sys.executable, '-m', 'autohar_researcher.main', 'search',
        '--manifest', str(manifest), '--protocol', 'lodo', '--target', 'all',
        '--method', 'fixed', '--baseline', str(baseline), '--seeds', '13',
        '--device', 'cpu', '--output', str(output), '--final-test'],
        capture_output=True, text=True, timeout=90)
    assert result.returncode == 0, result.stderr
    summary = json.loads((output / 'summary.json').read_text())
    assert len(summary['runs']) == 3
    assert set(summary['targets']) == set(value['datasets'])
    for target in value['datasets']:
        root = output / 'seed-13' / target
        split = json.loads((root / 'splits.json').read_text())
        assert {r[0] for r in split['test']} == {target}
        assert target not in {r[0] for r in split['train'] + split['validation']}
        assert (root / 'final-test.json').exists()
        assert 0 <= summary['targets'][target]['test_mean'] <= 1
        assert 'test_macro_f1' not in (root / 'history.jsonl').read_text()


def test_lodo_all_targets_split_preview(manifest):
    result = subprocess.run([sys.executable, '-m', 'autohar_researcher.main', 'splits',
        '--manifest', str(manifest), '--protocol', 'lodo', '--target', 'all'],
        capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    folds = json.loads(result.stdout)
    assert {f['target'] for f in folds} == {'uci_har', 'motion_sense', 'real_life_har'}
    assert all({r[0] for r in f['test']} == {f['target']} for f in folds)
