import json

import numpy as np
import pytest
import torch

from autohar_researcher.agents import Search
from autohar_researcher.datasets import make_task, windows
from autohar_researcher.harness import HARHarness
from autohar_researcher.preprocessing_artifacts import pipeline_id, snapshot_path
from autohar_researcher.types import Pipeline


def candidate(offset):
    p = Pipeline.from_dict({'data': {'window_seconds': 1, 'normalization': 'standard'},
        'model': {'architecture': 'cnn', 'hidden_dim': 4, 'depth': 1, 'max_epochs': 1}})
    p.code['research/components/data.py'] = f'''import numpy as np
def fit_preprocessor(x_train, params):
    return {{'center': np.mean(x_train, axis=(0, 1), keepdims=True) + {offset}, 'windows': len(x_train)}}
def transform(x, state):
    return x - state['center']
'''
    return p


@pytest.mark.parametrize('workers', [1, 2])
def test_candidates_keep_separate_recoverable_validation_snapshots(manifest, tmp_path, workers):
    torch.set_num_threads(1)
    task = make_task(manifest, 'loso', 'uci_har', '0')
    h = HARHarness(task, 13, tmp_path / f'workers-{workers}', 'cpu')
    h.candidate_workers, h.worker_threads = workers, 1
    candidates = [candidate(0), candidate(2)]
    search = Search(h, None, h.output, rounds=1, budget=2)
    selected, _ = search.run([{'id': str(i), 'pipeline': p.payload()} for i, p in enumerate(candidates)])
    snapshots, files = [], {}
    for p in candidates:
        path = snapshot_path(h.output, p)
        assert path.is_file()
        files[path] = path.read_bytes()
        saved = h.load_validation_preprocessing(p)
        snapshots.append(saved)
        assert saved['pipeline'] == p.payload()
        assert saved['pipeline_id'] == pipeline_id(p)
        assert saved['task_fingerprint'] == task.fingerprint and saved['seed'] == 13
        assert saved['data_identity'] == task.identity_metadata()
        assert saved['fit_scope'] == 'source_training_only'
        assert saved['artifact_role'] == 'candidate_validation_preprocessing'
        assert saved['state']['windows'] > 0
        np.testing.assert_array_equal(saved['state']['center'], h._preprocessing_states[h._key(p)]['center'])
        reference = h._validation_diagnostics[h._key(p)]['validation_preprocessing']
        assert reference['pipeline_id'] == saved['pipeline_id'] and reference['available']
        assert h.output / reference['path'] == path
        if workers == 2:
            worker_path = h.output / 'candidate-workers' / pipeline_id(p)[:16]
            worker_saved = torch.load(snapshot_path(worker_path, p), weights_only=False)
            np.testing.assert_array_equal(worker_saved['normalization_mean'], saved['normalization_mean'])
            np.testing.assert_array_equal(worker_saved['state']['center'], saved['state']['center'])
    assert pipeline_id(candidates[0]) != pipeline_id(candidates[1])
    np.testing.assert_allclose(snapshots[1]['state']['center'] - snapshots[0]['state']['center'], 2, atol=1e-6)
    assert not np.allclose(snapshots[0]['normalization_mean'], snapshots[1]['normalization_mean'])
    latest = torch.load(h.output / 'preprocessing-state.pt', weights_only=False)
    assert latest['pipeline_id'] == pipeline_id(candidates[-1])
    assert latest['artifact_role'] == 'latest_validation_evaluation_compatibility_copy'
    assert (h.output / latest['canonical_artifact']).is_file()
    selection = json.loads((h.output / 'selected.json').read_text())
    assert selection['validation_preprocessing']['pipeline_id'] == pipeline_id(selected)
    assert (h.output / selection['validation_preprocessing']['path']).is_file()
    # Recovery uses the candidate file even if a later candidate owns the latest alias.
    recovered = HARHarness(task, 13, h.output, 'cpu')
    for p, saved in zip(candidates, snapshots):
        np.testing.assert_array_equal(recovered.load_validation_preprocessing(p)['state']['center'], saved['state']['center'])
    # Final refitting must neither replace nor mutate either validation snapshot.
    h.finalize(selected)
    assert all(path.read_bytes() == before for path, before in files.items())
    final = torch.load(h.output / 'final-model.pt', weights_only=False)
    assert final['preprocessing_fit_scope'] == 'source_training_and_validation_final_refit'
    assert final['data_identity'] == task.identity_metadata()
    assert final['validation_preprocessing']['pipeline_id'] == pipeline_id(selected)
    combined, _, _ = windows(task.train + task.validation, task.labels, 1, selected.data['overlap'])
    offset = 0 if selected.payload() == candidates[0].payload() else 2
    np.testing.assert_allclose(final['preprocessing_state']['center'], combined.mean((0, 1), keepdims=True) + offset, rtol=1e-5)
    assert final['preprocessing_state']['windows'] > recovered.load_validation_preprocessing(selected)['state']['windows']


def test_second_sequential_evaluation_does_not_rewrite_first_candidate(manifest, tmp_path):
    torch.set_num_threads(1)
    h = HARHarness(make_task(manifest, 'loso', 'uci_har', '0'), 13, tmp_path / 'sequential', 'cpu')
    first, second = candidate(0), candidate(2)
    h.evaluate(first)
    path = snapshot_path(h.output, first)
    before = path.read_bytes()
    h.evaluate(second)
    assert path.read_bytes() == before


def test_legacy_snapshot_fallback_is_identity_checked_and_read_only(manifest, tmp_path):
    task = make_task(manifest, 'loso', 'uci_har', '0')
    h = HARHarness(task, 13, tmp_path / 'legacy', 'cpu')
    p = candidate(0)
    payload = dict(pipeline=p.payload(), state={'center': np.zeros((1,1,6))},
                   normalization_mean=np.zeros((1,1,6)), normalization_scale=np.ones((1,1,6)),
                   fit_scope='source_training_only')
    path = h.output / 'preprocessing-state.pt'
    torch.save(payload, path)
    before = path.read_bytes()
    saved = h.load_validation_preprocessing(p)
    assert saved['recovery_scope'] == 'legacy_snapshot_without_task_or_seed_provenance'
    assert path.read_bytes() == before and not snapshot_path(h.output, p).exists()
    with pytest.raises(ValueError, match='another pipeline'):
        h.load_validation_preprocessing(candidate(2))
    path.unlink()
    with pytest.raises(FileNotFoundError):
        h.load_validation_preprocessing(p)


def test_candidate_snapshot_rejects_wrong_context_and_never_falls_back(manifest, tmp_path):
    task = make_task(manifest, 'loso', 'uci_har', '0')
    h = HARHarness(task, 13, tmp_path / 'context', 'cpu')
    p = candidate(0)
    payload = dict(pipeline=p.payload(), state=None, normalization_mean=0., normalization_scale=1.)
    h._save_validation_preprocessing(p, payload)
    for seed, other_task in [(14, task), (13, make_task(manifest, 'loso', 'uci_har', '1'))]:
        other = HARHarness(other_task, seed, h.output, 'cpu')
        with pytest.raises(ValueError, match='context mismatch'):
            other.load_validation_preprocessing(p)
    canonical = snapshot_path(h.output, p)
    bad = torch.load(canonical, weights_only=False)
    bad['pipeline_id'] = 'wrong'
    torch.save(bad, canonical)
    with pytest.raises(ValueError, match='identifier mismatch'):
        h.load_validation_preprocessing(p)
    bad['pipeline_id'] = pipeline_id(p)
    bad['fit_scope'] = 'source_training_and_validation_final_refit'
    torch.save(bad, canonical)
    with pytest.raises(ValueError, match='source training only'):
        h.load_validation_preprocessing(p)
    with pytest.raises(ValueError, match='Cannot relabel'):
        h._save_validation_preprocessing(p, bad)


@pytest.mark.parametrize('workers', [1, 2])
def test_stateless_state_can_be_transferred_after_validation_and_final_refit(manifest, tmp_path, workers):
    """Reproduce the UTD handoff: no custom state must not cause a KeyError."""
    from autohar_researcher.parallel import evaluate_many
    torch.set_num_threads(1)
    task = make_task(manifest, 'loso', 'uci_har', '0')
    h = HARHarness(task, 13, tmp_path / 'original', 'cpu')
    p = candidate(0)
    p.code.clear()
    if workers == 1:
        result = h.evaluate(p)
    else:
        result = next(evaluate_many(h, [p], workers, 1))
        assert not isinstance(result, Exception)
    key = h._key(p)
    assert h._preprocessing_states[key] is None
    snapshot = h.load_validation_preprocessing(p)
    assert snapshot['state'] is None
    h.finalize(p)
    assert h._preprocessing_states[key] is None
    final = torch.load(h.output / 'final-model.pt', weights_only=False)
    assert final['preprocessing_state'] is None
    restored = HARHarness(task, 13, tmp_path / 'restored', 'cpu')
    restored._preprocessing_states[key] = h._preprocessing_states[key]
    restored._save_validation_preprocessing(p, h.load_validation_preprocessing(p))
    restored._evaluations[key] = list(h._evaluations[key])
    restored._lr_schedules[key] = list(h._lr_schedules[key])
    restored.finalize(p)
    assert restored.load_validation_preprocessing(p)['state'] is None


@pytest.mark.parametrize('missing', ['pipeline', 'state', 'normalization_mean', 'normalization_scale'])
def test_malformed_snapshot_is_rejected_before_evidence_transfer(manifest, tmp_path, missing):
    h = HARHarness(make_task(manifest, 'loso', 'uci_har', '0'), 13, tmp_path / 'malformed', 'cpu')
    p = candidate(0)
    payload = dict(pipeline=p.payload(), state=None, normalization_mean=0., normalization_scale=1.)
    h._save_validation_preprocessing(p, payload)
    path = snapshot_path(h.output, p)
    broken = torch.load(path, weights_only=False)
    del broken[missing]
    torch.save(broken, path)
    with pytest.raises(ValueError, match='Incomplete validation preprocessing snapshot'):
        h.load_validation_preprocessing(p)
