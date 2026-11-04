import json
import os

import numpy as np
import pytest

from autohar_researcher.datasets import load_dataset_snapshot, make_task, make_fixed_validation_task, windows
from autohar_researcher.loso import LOSOHarness


def task_for(path, protocol, **kwargs):
    if protocol == 'fixed-validation':
        return make_fixed_validation_task(path, 'uci_har', '0', **kwargs)
    return make_task(path, 'loso', 'uci_har', '0', **kwargs)


@pytest.mark.parametrize('protocol,partition', [('loso','train'), ('loso','validation'), ('loso','test'),
                                               ('fixed-validation','train'), ('fixed-validation','validation')])
def test_content_change_changes_identity_even_with_same_manifest_size_and_mtime(manifest, protocol, partition):
    before = task_for(manifest, protocol)
    manifest_bytes = manifest.read_bytes()
    recording = getattr(before, partition)[0]
    signature = recording.path.stat()
    original = np.load(recording.path, allow_pickle=False)
    original[0, 0] += 1.
    np.save(recording.path, original, allow_pickle=False)
    os.utime(recording.path, ns=(signature.st_atime_ns, signature.st_mtime_ns))
    assert recording.path.stat().st_size == signature.st_size
    assert recording.path.stat().st_mtime_ns == signature.st_mtime_ns
    assert manifest.read_bytes() == manifest_bytes
    after = task_for(manifest, protocol)
    assert before.data_fingerprint != after.data_fingerprint
    assert before.fingerprint != after.fingerprint
    assert before.manifest_sha256 == after.manifest_sha256
    assert before.split_fingerprint == after.split_fingerprint
    assert task_for(manifest, protocol).fingerprint == after.fingerprint


@pytest.mark.parametrize('protocol', ['loso', 'fixed-validation'])
def test_unchanged_data_stable_and_split_seed_separate(manifest, protocol):
    first = task_for(manifest, protocol, split_seed=13)
    second = task_for(manifest, protocol, split_seed=14)
    assert first.data_fingerprint == second.data_fingerprint
    assert first.split_fingerprint != second.split_fingerprint
    assert first.fingerprint != second.fingerprint
    assert task_for(manifest, protocol, split_seed=13).split_manifest() == first.split_manifest()
    assert first.fingerprint_scheme == 'recording-content-and-split-v1'


def test_grouping_changes_split_but_not_recording_identity(manifest):
    original = make_task(manifest, 'loso', 'uci_har', '0')
    original_fixed = make_fixed_validation_task(manifest, 'uci_har', '0')
    raw = json.loads(manifest.read_text())
    raw['datasets']['uci_har']['loso_groups'] = [['0','1'], ['2','3'], ['4','5']]
    manifest.write_text(json.dumps(raw))
    grouped = make_task(manifest, 'loso', 'uci_har', '0')
    fixed = make_fixed_validation_task(manifest, 'uci_har', '0')
    assert grouped.data_fingerprint == original.data_fingerprint
    assert grouped.split_fingerprint != original.split_fingerprint
    assert grouped.fingerprint != original.fingerprint
    assert grouped.manifest_sha256 != original.manifest_sha256
    assert fixed.data_fingerprint == original_fixed.data_fingerprint
    assert fixed.split_fingerprint == original_fixed.split_fingerprint


def test_reordering_and_formatting_manifest_preserves_semantic_identity(manifest):
    first = make_task(manifest, 'loso', 'uci_har', '0')
    raw = json.loads(manifest.read_text())
    raw['recordings'].reverse()
    manifest.write_text(json.dumps(raw, indent=2))
    second = make_task(manifest, 'loso', 'uci_har', '0')
    assert first.manifest_sha256 != second.manifest_sha256
    assert first.data_fingerprint == second.data_fingerprint
    assert first.split_fingerprint == second.split_fingerprint
    # The composite also retains the original manifest-byte sensitivity, since
    # manifest record ordering can affect training iteration order.
    assert first.fingerprint != second.fingerprint


def test_snapshot_hashes_once_for_folds_and_never_in_window_training(manifest, monkeypatch):
    from autohar_researcher import datasets
    calls = []
    checksum = datasets.recording_checksum
    def track(path):
        calls.append(path)
        return checksum(path)
    monkeypatch.setattr(datasets, 'recording_checksum', track)
    snapshot = load_dataset_snapshot(manifest)
    count = len(snapshot.recordings)
    assert len(calls) == count
    tasks = [make_task(manifest, 'loso', 'uci_har', str(i), snapshot=snapshot) for i in range(6)]
    fixed = make_fixed_validation_task(manifest, 'uci_har', '0', snapshot=snapshot)
    windows(tasks[0].train, tasks[0].labels, 1, 0)
    windows(fixed.train, fixed.labels, 1, 0)
    assert len(calls) == count
    snapshot_again = load_dataset_snapshot(manifest)
    assert len(calls) == 2*count  # No timestamp-only cache across run setups.
    assert snapshot_again.data_fingerprint == snapshot.data_fingerprint


def test_snapshot_ledger_contains_actual_streamed_file_hashes(manifest, monkeypatch):
    import hashlib
    monkeypatch.setattr(np, 'load', lambda *args, **kwargs: pytest.fail('Identity must not interpret signal arrays'))
    snapshot = load_dataset_snapshot(manifest)
    provenance = snapshot.provenance()
    assert provenance['data_fingerprint'] == snapshot.data_fingerprint
    assert provenance['checksum_algorithm'] == 'sha256'
    for row in provenance['recordings']:
        path = manifest.parent / row['path']
        assert row['sha256'] == hashlib.sha256(path.read_bytes()).hexdigest()
        assert row['size_bytes'] == path.stat().st_size


def test_stale_run_snapshot_fails_instead_of_training_on_changed_data(manifest):
    snapshot = load_dataset_snapshot(manifest)
    task = make_task(manifest, 'loso', 'uci_har', '0', snapshot=snapshot)
    recording = task.train[0]
    stat = recording.path.stat()
    os.utime(recording.path, ns=(stat.st_atime_ns, stat.st_mtime_ns+1000000000))
    with pytest.raises(ValueError, match='changed after dataset identity verification'):
        recording.load()
    with pytest.raises(ValueError, match='changed after dataset identity verification'):
        make_task(manifest, 'loso', 'uci_har', '1', snapshot=snapshot)


def test_grouped_tasks_reject_mixed_data_versions(manifest, tmp_path):
    first = make_task(manifest, 'loso', 'uci_har', '0')
    recording = first.train[0]
    values = np.load(recording.path)
    values[0,0] += 1
    np.save(recording.path, values)
    second = make_task(manifest, 'loso', 'uci_har', '1')
    with pytest.raises(ValueError, match='same verified recording data identity'):
        LOSOHarness([first, second], 13, tmp_path/'mixed', 'cpu', allow_partial_folds=True)


def test_cli_split_preview_shares_one_verified_snapshot(manifest, monkeypatch, capsys):
    import sys
    from autohar_researcher import datasets, main
    calls = []
    original = datasets.recording_checksum
    def track(path):
        calls.append(path)
        return original(path)
    monkeypatch.setattr(datasets, 'recording_checksum', track)
    monkeypatch.setattr(sys, 'argv', ['autohar', 'splits', '--manifest', str(manifest),
                                    '--protocol','loso','--target','uci_har'])
    main.main()
    folds = json.loads(capsys.readouterr().out)
    assert len(folds) == 6
    assert len(calls) == len(json.loads(manifest.read_text())['recordings'])
    assert len({row['data_fingerprint'] for row in folds}) == 1
    assert len({row['split_fingerprint'] for row in folds}) == 6
