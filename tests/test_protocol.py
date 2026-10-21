import dataclasses
import json

import numpy as np
import pytest

from autohar_researcher.datasets import make_task, windows
from autohar_researcher.harness import Normalizer, domain_scores, macro_f1


def subjects(rows):
    return {(r.dataset, r.subject) for r in rows}


def test_loso_subject_disjoint_and_stable(manifest):
    task = make_task(manifest, 'loso', 'uci_har', '0')
    assert not subjects(task.train) & subjects(task.validation)
    assert not subjects(task.train + task.validation) & subjects(task.test)
    assert subjects(task.test) == {('uci_har', '0')}
    assert len(subjects(task.validation)) == 1
    assert make_task(manifest, 'loso', 'uci_har', '0').split_manifest() == task.split_manifest()
    assert 'path' not in json.dumps(task.public_metadata())


def test_lodo_target_is_not_a_source(manifest):
    task = make_task(manifest, 'lodo', 'uci_har')
    assert {r.dataset for r in task.train} == {'motion_sense', 'real_life_har'}
    assert {r.dataset for r in task.validation} == {'motion_sense', 'real_life_har'}
    assert {r.dataset for r in task.test} == {'uci_har'}
    assert not subjects(task.train) & subjects(task.validation)
    with pytest.raises(dataclasses.FrozenInstanceError):
        task.target = 'motion_sense'


def test_split_building_does_not_open_signal_arrays(manifest, monkeypatch):
    monkeypatch.setattr(np, 'load', lambda *a, **k: pytest.fail('signal access during task initialization'))
    make_task(manifest, 'lodo', 'uci_har')


def test_reject_invalid_lodo_target(manifest):
    with pytest.raises(ValueError, match='LODO targets'):
        make_task(manifest, 'lodo', 'hapt')


def test_fixed_classes_include_missing_classes_in_metric():
    assert macro_f1(np.array([0, 0]), np.array([0, 0]), 2) == .5


def test_equal_domain_weight_not_pooled_metric():
    y = np.array([0] * 100 + [1])
    pred = np.array([0] * 101)
    scores = domain_scores(pred, y, np.array(['a'] * 100 + ['b']), 2)
    assert np.mean(list(scores.values())) == .25
    assert macro_f1(pred, y, 2) != .25


def test_normalizer_never_fits_validation():
    train = np.array([[[0., 2.], [2., 4.]]])
    normalizer = Normalizer('standard').fit(train)
    before = normalizer.mean.copy()
    transformed = normalizer.transform(np.full((1, 2, 2), 1000.))
    np.testing.assert_array_equal(normalizer.mean, before)
    assert transformed.min() > 900


def test_windowing_preserves_sessions(manifest):
    task = make_task(manifest, 'loso', 'uci_har', '0')
    x, y, domains = windows(task.train, task.labels, 2, 0)
    assert x.shape[1:] == (100, 6)
    assert len(x) == len(task.train)
    with pytest.raises(ValueError):
        windows(task.train, task.labels, 6, 0)


def test_polyphase_fixed_50hz(manifest):
    task = make_task(manifest, 'loso', 'uci_har', '0')
    recording = dataclasses.replace(task.train[0], hz=100)
    assert recording.load().shape == (50, 6)


def test_duplicate_recording_cannot_leak_between_subjects(manifest):
    value = json.loads(manifest.read_text())
    value['recordings'].append(dict(value['recordings'][0], subject='99'))
    manifest.write_text(json.dumps(value))
    with pytest.raises(ValueError, match='only once'):
        make_task(manifest, 'loso', 'uci_har', '0')


def test_loso_cycles_every_person_in_both_held_out_roles(manifest):
    from collections import Counter
    test_counts, val_counts = Counter(), Counter()
    people = {str(i) for i in range(6)}
    for person in people:
        task = make_task(manifest, 'loso', 'uci_har', person)
        train = {r.subject for r in task.train}
        val = {r.subject for r in task.validation}
        test = {r.subject for r in task.test}
        assert test == {person}
        assert len(val) == 1
        assert len(train) == 4
        assert not (train & val or train & test or val & test)
        assert train | val | test == people
        test_counts.update(test)
        val_counts.update(val)
    assert test_counts == val_counts == Counter({p: 1 for p in people})


def test_loso_one_validation_person_even_with_many_people(manifest):
    value = json.loads(manifest.read_text())
    template = value['recordings'][0]
    for person in range(6, 20):
        value['recordings'].append(dict(template, subject=str(person),
            session=f'{person}-0', path=f'unopened-{person}.npy'))
    manifest.write_text(json.dumps(value))
    task = make_task(manifest, 'loso', 'uci_har', '0')
    assert len(subjects(task.validation)) == 1
    assert len(subjects(task.train)) == 18


def test_loso_needs_three_people(manifest):
    value = json.loads(manifest.read_text())
    value['recordings'] = [r for r in value['recordings'] if r['subject'] in {'0', '1'}]
    manifest.write_text(json.dumps(value))
    with pytest.raises(ValueError, match='at least three'):
        make_task(manifest, 'loso', 'uci_har', '0')
