import json

import numpy as np
import pytest
import torch

from autohar_researcher.datasets import make_task
from autohar_researcher.loso import LOSOHarness
from autohar_researcher.types import Evaluation, Pipeline


def harness(manifest, tmp_path):
    return LOSOHarness([make_task(manifest, 'loso', 'uci_har', str(i)) for i in range(6)],
                       13, tmp_path / 'dataset', device='cpu')


def test_partial_folds_need_opt_in_and_keep_all_source_subjects(manifest, tmp_path, monkeypatch):
    tasks = [make_task(manifest, 'loso', 'uci_har', str(i)) for i in range(2)]
    with pytest.raises(ValueError, match='every subject'):
        LOSOHarness(tasks, 13, tmp_path / 'strict', device='cpu')
    h = LOSOHarness(tasks, 13, tmp_path / 'partial', device='cpu', allow_partial_folds=True)
    assert h.subjects == ['0', '1']
    assert h.available_subject_count == 6
    for subject, fold in zip(h.subjects, h.folds):
        assert {r.subject for r in fold.task.train + fold.task.validation} == set('012345') - {subject}
    p = Pipeline()
    key = h._key(p)
    for index, fold in enumerate(h.folds):
        def evaluate(p, index=index, fold=fold):
            fold._evaluations[key] = [1]
            fold._lr_schedules[key] = [.001]
            return Evaluation(.2 + .4 * index, {'uci_har': .2 + .4 * index}, 1)
        monkeypatch.setattr(fold, 'evaluate', evaluate)
        monkeypatch.setattr(fold, 'finalize', lambda p: {'test_macro_f1': .5})
    assert h.evaluate(p).score == pytest.approx(.4)
    report = h.finalize(p)
    assert report['tested_subjects'] == ['0', '1']
    assert report['available_subject_count'] == 6


def test_only_mean_validation_drives_selection_and_fold_epochs_are_retained(manifest, tmp_path, monkeypatch):
    h = harness(manifest, tmp_path)
    p = Pipeline()
    key = h._key(p)
    for index, fold in enumerate(h.folds):
        def evaluate(p, fold=fold, index=index):
            fold._evaluations[key] = [index + 1]
            fold._lr_schedules[key] = [.001] * (index + 1)
            return Evaluation(index / 10, {'uci_har': index / 10}, index + 1)
        monkeypatch.setattr(fold, 'evaluate', evaluate)
        def finalize(p, fold=fold):
            return {'test_macro_f1': .5, 'refit_epochs': fold._evaluations[key][0]}
        monkeypatch.setattr(fold, 'finalize', finalize)
    result = h.evaluate(p)
    assert result.score == pytest.approx(.25)
    assert result.dataset_scores == {'uci_har': result.score}
    assert h._validation_diagnostics[key]['mean_validation_macro_f1'] == result.score
    assert len(h._validation_diagnostics[key]['folds']) == 6
    report = h.finalize(p)
    assert report['test_std_across_folds'] == 0
    assert 'test_std_across_subjects' not in report
    assert 'exploratory' in report['evaluation_scope']
    assert [r['refit_epochs'] for r in report['final_loso']] == list(range(1, 7))
    with pytest.raises(RuntimeError, match='closed'):
        h.evaluate(p)
    with pytest.raises(RuntimeError, match='once'):
        h.finalize(p)


def test_failed_fold_never_produces_candidate_score(manifest, tmp_path, monkeypatch):
    h = harness(manifest, tmp_path)
    p = Pipeline()
    monkeypatch.setattr(h.folds[0], 'evaluate', lambda p: (_ for _ in ()).throw(ValueError('bad fold')))
    with pytest.raises(ValueError, match='bad fold'):
        h.evaluate(p)
    assert not h._evaluations and not h._evidence
    with pytest.raises(ValueError, match='every fold'):
        h.finalize(p)


def test_real_dataset_search_respects_each_folds_exclusion_and_refits(manifest, tmp_path, monkeypatch):
    torch.set_num_threads(1)
    h = harness(manifest, tmp_path)
    p = Pipeline.from_dict({'data': {'window_seconds': 1},
        'model': {'architecture': 'cnn', 'depth': 1, 'hidden_dim': 4, 'max_epochs': 1}})
    original_load = np.load
    accessed = set()
    def load(path, *args, **kwargs):
        accessed.add(path)
        return original_load(path, *args, **kwargs)
    monkeypatch.setattr(np, 'load', load)
    for fold in h.folds:
        evaluate = fold.evaluate
        def checked(p, evaluate=evaluate, fold=fold):
            accessed.clear()
            result = evaluate(p)
            assert not accessed & {r.path for r in fold.task.test}
            return result
        monkeypatch.setattr(fold, 'evaluate', checked)
    result = h.evaluate(p)
    assert 0 <= result.score <= 1
    report = h.finalize(p)
    assert len(report['final_loso']) == 6
    assert all(r['refit_epochs'] == 1 for r in report['final_loso'])
    for fold in h.folds:
        events = [json.loads(line) for line in (fold.output / 'training-progress.jsonl').read_text().splitlines()]
        assert len([e for e in events if e['stage'] == 'final_refit']) == 1
    evidence = h.selection_evidence(p)
    assert len(evidence['folds']) == 6
    assert evidence['elapsed_seconds'] > 0


def test_parallel_candidates_keep_fold_evidence(manifest, tmp_path):
    torch.set_num_threads(1)
    h = harness(manifest, tmp_path)
    pipelines = [Pipeline.from_dict({'data': {'window_seconds': 1},
        'model': {'architecture': 'cnn', 'depth': 1, 'hidden_dim': width, 'max_epochs': 1}})
        for width in (4, 8)]
    results = list(h.evaluate_many(pipelines, workers=2, threads=1))
    assert all(isinstance(r, Evaluation) for r in results), results
    for pipeline in pipelines:
        evidence = h.selection_evidence(pipeline)
        assert len(evidence['folds']) == 6
        assert evidence['batch_elapsed_seconds'] > 0
        assert all(f['epochs'] == [1] and len(f['learning_rates']) == 1 for f in evidence['folds'])
    assert len(h.finalize(pipelines[0])['final_loso']) == 6
