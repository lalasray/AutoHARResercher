from copy import deepcopy
import json

import numpy as np
import pytest
import torch

from autohar_researcher.datasets import make_task, windows
from autohar_researcher.evaluation_reporting import load_loso_report, loso_test_statistics, normalize_loso_report
from autohar_researcher.harness import HARHarness, macro_f1
from autohar_researcher.loso import LOSOHarness
from autohar_researcher.types import Pipeline


def unequal_reports():
    # Pooled fold scores need not equal the mean of individual subject F1s.
    return [
        {'subject': '0', 'test_subjects': ['0'], 'test_macro_f1': .9, 'test_subject_scores': {'0': .9}},
        {'subject': '1', 'test_subjects': ['1', '2'], 'test_macro_f1': .4, 'test_subject_scores': {'1': .1, '2': .7}},
        {'subject': '3', 'test_subjects': ['3', '4', '5'], 'test_macro_f1': .6, 'test_subject_scores': {'3': .1, '4': .2, '5': .8}},
    ]


def test_unequal_groups_have_explicit_distinct_statistics(manifest, tmp_path, monkeypatch):
    rows = unequal_reports()
    raw = json.loads(manifest.read_text())
    raw['datasets']['uci_har']['loso_groups'] = [r['test_subjects'] for r in rows]
    manifest.write_text(json.dumps(raw))
    h = LOSOHarness([make_task(manifest, 'loso', 'uci_har', r['subject']) for r in rows],
                    13, tmp_path / 'loso', 'cpu')
    p = Pipeline()
    h._evidence[h._key(p)] = {'already_selected': True}
    for fold, row in zip(h.folds, rows):
        monkeypatch.setattr(fold, 'finalize', lambda pipeline, row=row: deepcopy(row))
    report = h.finalize(p)
    fold_values = [.9, .4, .6]
    subject_values = [.9, .1, .7, .1, .2, .8]
    assert report['test_macro_f1'] == pytest.approx(np.mean(fold_values))
    assert report['test_std_across_folds'] == pytest.approx(np.std(fold_values, ddof=1))
    stats = report['test_subject_statistics']
    assert stats['status'] == 'complete' and stats['scored_subject_count'] == stats['expected_subject_count'] == 6
    assert stats['mean'] == pytest.approx(np.mean(subject_values))
    assert stats['std'] == pytest.approx(np.std(subject_values, ddof=1))
    assert stats['mean'] != pytest.approx(report['test_macro_f1'])
    assert report['test_macro_f1'] != pytest.approx(np.average(fold_values, weights=[1, 2, 3]))
    assert 'equal' in report['test_metric_scope']['fold_weighting']
    assert 'test_std_across_subjects' not in report
    assert json.loads((h.output / 'final-test.json').read_text()) == report
    with pytest.raises(RuntimeError, match='once'):
        h.finalize(p)


def test_legacy_grouped_artifact_is_normalized_without_mutation_or_invented_subjects(tmp_path):
    rows = unequal_reports()
    for row in rows:
        row.pop('test_subject_scores')
    original = {'test_macro_f1': float(np.mean([.9, .4, .6])),
                'test_std_across_subjects': float(np.std([.9, .4, .6], ddof=1)),
                'final_loso': rows, 'tested_subjects': list('012345'), 'custom_metadata': 'keep'}
    path = tmp_path / 'legacy.json'
    path.write_text(json.dumps(original))
    before = path.read_bytes()
    result = load_loso_report(path)
    assert path.read_bytes() == before
    assert 'test_std_across_subjects' in original
    assert 'test_std_across_subjects' not in result
    assert result['test_macro_f1'] == original['test_macro_f1']
    assert result['final_loso'] == rows and result['custom_metadata'] == 'keep'
    assert result['test_std_across_folds'] == original['test_std_across_subjects']
    assert result['test_subject_statistics']['mean'] is None
    assert result['test_subject_statistics']['std'] is None
    assert result['test_subject_statistics']['status'] == 'unavailable'
    assert result['test_subject_statistics']['missing_subjects'] == list('012345')
    assert normalize_loso_report(result) == result


def test_legacy_unknown_membership_does_not_treat_representative_as_individual_score():
    legacy = {'test_macro_f1': .5, 'test_std_across_subjects': 0.,
              'final_loso': [{'subject': '0', 'test_macro_f1': .5}]}
    result = normalize_loso_report(legacy)
    stats = result['test_subject_statistics']
    assert not stats['fold_membership_known'] and stats['mean'] is None
    assert result['test_std_across_folds'] == 0.


def test_single_fold_and_individual_subject_scopes():
    report = loso_test_statistics(unequal_reports()[:1])
    assert report['tested_fold_count'] == 1
    assert report['test_std_across_folds'] == 0.
    assert report['test_subject_statistics']['std'] == 0.
    assert report['test_subject_statistics']['mean'] == report['test_macro_f1']
    assert report['test_metric_scope']['std_ddof'] == 1


def test_partial_subject_evidence_is_not_summarized_as_all_subjects():
    rows = unequal_reports()
    rows[1]['test_subject_scores'].pop('2')
    stats = loso_test_statistics(rows)['test_subject_statistics']
    assert stats['status'] == 'unavailable'
    assert stats['missing_subjects'] == ['2'] and stats['scored_subject_count'] == 5
    assert stats['expected_subject_count'] == 6
    assert stats['mean'] is None and stats['std'] is None


def test_windows_optional_subject_ids_preserve_original_arrays(manifest):
    task = make_task(manifest, 'loso', 'uci_har', '0')
    original = windows(task.test, task.labels, 1, .5)
    detailed = windows(task.test, task.labels, 1, .5, return_subjects=True)
    for a, b in zip(original, detailed[:3]):
        np.testing.assert_array_equal(a, b)
    assert set(detailed[3]) == {'0'} and len(detailed[3]) == len(original[1])


def test_per_subject_final_scores_use_one_prediction_and_fit_with_no_target_selection(manifest, tmp_path, monkeypatch):
    torch.set_num_threads(1)
    raw = json.loads(manifest.read_text())
    raw['datasets']['uci_har']['loso_groups'] = [['0', '1'], ['2', '3'], ['4', '5']]
    manifest.write_text(json.dumps(raw))
    task = make_task(manifest, 'loso', 'uci_har', '0')
    h = HARHarness(task, 13, tmp_path / 'one-pass', 'cpu')
    p = Pipeline.from_dict({'data': {'window_seconds': 1},
        'model': {'architecture': 'cnn', 'hidden_dim': 4, 'depth': 1, 'max_epochs': 1}})
    original_load = np.load
    accessed = set()
    def track_load(path, *args, **kwargs):
        accessed.add(path)
        return original_load(path, *args, **kwargs)
    monkeypatch.setattr(np, 'load', track_load)
    h.evaluate(p)
    assert not accessed.intersection(r.path for r in task.test)
    stored_selection = deepcopy(h._evaluations)
    original_fit, original_predict = h._fit, h._predict
    fit_calls, predictions = [], []
    def fit(*args, **kwargs):
        fit_calls.append(kwargs)
        assert kwargs.get('validation') is None
        return original_fit(*args, **kwargs)
    def predict(*args, **kwargs):
        value = original_predict(*args, **kwargs)
        predictions.append(value)
        return value
    monkeypatch.setattr(h, '_fit', fit)
    monkeypatch.setattr(h, '_predict', predict)
    report = h.finalize(p)
    assert len(fit_calls) == len(predictions) == 1
    assert h._evaluations == stored_selection
    _, y, _, subjects = windows(task.test, task.labels, 1, p.data['overlap'], return_subjects=True)
    assert report['test_macro_f1'] == macro_f1(predictions[0], y, len(task.labels))
    for person in ('0', '1'):
        mask = subjects == person
        assert report['test_subject_scores'][person] == macro_f1(predictions[0][mask], y[mask], len(task.labels))
        assert report['test_subject_window_counts'][person] == int(mask.sum())
    assert json.loads((h.output / 'final-test.json').read_text()) == report
    with pytest.raises(RuntimeError, match='once'):
        h.finalize(p)
    with pytest.raises(RuntimeError, match='closed'):
        h.evaluate(p)
