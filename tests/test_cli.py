import json
import subprocess
import sys


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
    baseline = tmp_path / 'baseline.json'
    baseline.write_text(json.dumps({'model': {'architecture': 'cnn', 'hidden_dim': 4,
        'depth': 1, 'max_epochs': 1}, 'data': {'window_seconds': 1}}))
    output = tmp_path / 'cli'
    result = subprocess.run([sys.executable, '-m', 'autohar_researcher.main', 'search',
            '--manifest', str(manifest), '--protocol', 'loso', '--target', 'uci_har',
            '--subject', '0', '--seeds', '13', '--rounds', '1', '--budget', '2',
            '--device', 'cpu', '--baseline', str(baseline), '--output', str(output), '--final-test',
            '--backend', 'codex', '--codex-bin', str(backend), '--model', 'test'],
            capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr
    summary = json.loads((output / 'summary.json').read_text())
    assert summary['runs'][0]['evaluations'] == 2
    assert 0 <= summary['test_mean'] <= 1
    history = (output / 'seed-13/subject-0/history.jsonl').read_text()
    assert 'test_macro_f1' not in history


def test_loso_cli_defaults_to_complete_person_cycle(manifest):
    result = subprocess.run([sys.executable, '-m', 'autohar_researcher.main', 'splits',
        '--manifest', str(manifest), '--protocol', 'loso', '--target', 'uci_har'],
        capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    folds = json.loads(result.stdout)
    assert len(folds) == 6
    test_people = [next(iter({row[1] for row in fold['test']})) for fold in folds]
    val_people = [next(iter({row[1] for row in fold['validation']})) for fold in folds]
    assert sorted(test_people) == sorted(val_people) == list('012345')
    assert all(test != val for test, val in zip(test_people, val_people))
