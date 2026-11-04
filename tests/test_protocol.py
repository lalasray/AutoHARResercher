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


def test_loso_holds_out_every_person_once(manifest):
    from collections import Counter
    test_counts = Counter()
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
    assert test_counts == Counter({p: 1 for p in people})


def test_loso_80_20_source_subject_split(manifest):
    value = json.loads(manifest.read_text())
    template = value['recordings'][0]
    for person in range(6, 20):
        np.save(manifest.parent / f'unopened-{person}.npy', np.zeros((100, 6), dtype=np.float32))
        value['recordings'].append(dict(template, subject=str(person),
            session=f'{person}-0', path=f'unopened-{person}.npy'))
    manifest.write_text(json.dumps(value))
    task = make_task(manifest, 'loso', 'uci_har', '0')
    assert len(subjects(task.validation)) == 4
    assert len(subjects(task.train)) == 15
    assert not subjects(task.test) & subjects(task.train + task.validation)


def test_loso_needs_three_people(manifest):
    value = json.loads(manifest.read_text())
    value['recordings'] = [r for r in value['recordings'] if r['subject'] in {'0', '1'}]
    manifest.write_text(json.dumps(value))
    with pytest.raises(ValueError, match='at least three'):
        make_task(manifest, 'loso', 'uci_har', '0')


@pytest.mark.parametrize('target', ['uci_har', 'motion_sense', 'real_life_har'])
def test_lodo_all_targets_with_three_class_real_life(manifest, target):
    value = json.loads(manifest.read_text())
    supported = ['walking', 'sitting', 'standing']
    value['datasets']['real_life_har']['labels'] = supported
    value['recordings'] = [r for r in value['recordings']
        if r['dataset'] != 'real_life_har' or r['label'] in {'0', '3', '4'}]
    manifest.write_text(json.dumps(value))
    task = make_task(manifest, 'lodo', target)
    assert len(task.labels) == 5
    assert {r.dataset for r in task.test} == {target}
    assert {r.dataset for r in task.train} == set(value['datasets']) - {target}
    assert not subjects(task.train) & subjects(task.validation)
    for name in set(value['datasets']) - {target}:
        assert len({r.subject for r in task.validation if r.dataset == name}) == 2

    assert macro_f1(np.array([0, 3, 4]), np.array([0, 3, 4]), 5) == pytest.approx(.6)


def test_lodo_requires_complete_manifest(manifest):
    value = json.loads(manifest.read_text())
    del value['datasets']['real_life_har']
    value['recordings'] = [r for r in value['recordings'] if r['dataset'] != 'real_life_har']
    manifest.write_text(json.dumps(value))
    with pytest.raises(ValueError, match='all three datasets'):
        make_task(manifest, 'lodo', 'uci_har')


def test_real_life_loso_defaults_to_three_classes(manifest):
    from autohar_researcher.datasets import make_fixed_validation_task

    task = make_task(manifest, 'loso', 'real_life_har', '0')
    assert task.labels == ('sitting', 'standing', 'walking')
    for partition in (task.train, task.validation, task.test):
        assert {r.label for r in partition} == set(task.labels)
    fixed = make_fixed_validation_task(manifest, 'real_life_har', '0')
    assert fixed.labels == task.labels
    assert {r.label for r in fixed.train + fixed.validation} == set(task.labels)
    assert len(make_task(manifest, 'lodo', 'real_life_har').labels) == 5
    assert macro_f1(np.arange(3), np.arange(3), len(task.labels)) == 1.0


@pytest.mark.parametrize('target,count', [('wisdm', 6), ('hang_time', 9), ('capture_24', 12), ('utd_mhad', 11)])
def test_requested_loso_class_spaces_are_fixed(manifest, target, count):
    from autohar_researcher.benchmark_data import LOSO_LABELS
    value = json.loads(manifest.read_text())
    spec = value['datasets'].pop('uci_har')
    labels = LOSO_LABELS[target]
    spec['labels'] = list(labels) + ['excluded']
    spec['label_map'] = {str(i): labels[i] for i in range(5)}
    value['datasets'][target] = spec
    for row in value['recordings']:
        if row['dataset'] == 'uci_har':
            row['dataset'] = target
    manifest.write_text(json.dumps(value))
    task = make_task(manifest, 'loso', target, '0')
    assert len(task.labels) == count
    assert 'excluded' not in task.labels
    assert not subjects(task.train) & subjects(task.validation)
    assert not subjects(task.train + task.validation) & subjects(task.test)
