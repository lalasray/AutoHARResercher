import copy
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from autohar_researcher.agents import Search, SingleAgentSearch
from autohar_researcher.baselines import dataset_candidates, initial_candidates, registry
from autohar_researcher.harness import Normalizer
from autohar_researcher.patches import validate_pipeline
from autohar_researcher.types import Pipeline, LOSO_DATASETS
from test_search import Harness, Agents


def candidates():
    result = []
    for i in range(3):
        p = Pipeline.from_dict({'model': {'architecture': 'cnn', 'hidden_dim': 4+i}})
        result.append({'id': str(i), 'pipeline': p.payload(), 'evidence': {'fixture': i}})
    return result


@pytest.mark.parametrize('search_type', [Search, SingleAgentSearch])
def test_validation_selects_start_and_charges_budget_without_retraining(tmp_path, search_type):
    h = Harness([.4, .8, .8])
    search = search_type(h, Agents(), tmp_path, budget=3)
    selected, score = search.run(candidates())
    assert search.evaluated == len(h.calls) == 3
    assert selected.model['hidden_dim'] == 5 and score.score == .8
    report = json.loads((tmp_path/'initialization.json').read_text())
    assert report['selected_id'] == '1'
    assert all(row['status'] == 'evaluated' for row in report['candidates'])
    assert search.starting_point_evidence == {'fixture': 1}


def test_preflight_rejection_and_failed_fit_accounting(tmp_path):
    h = Harness([.7])
    def preflight(p):
        if p.model['hidden_dim'] == 4:
            raise ValueError('shape')
    evaluate = h.evaluate
    def fit(p):
        if p.model['hidden_dim'] == 5:
            raise RuntimeError('training failed')
        return evaluate(p)
    h.preflight, h.evaluate = preflight, fit
    search = Search(h, Agents(), tmp_path, budget=3)
    selected, score = search._initialize(candidates())
    assert search.evaluated == 2 and selected.model['hidden_dim'] == 6
    rows = json.loads((tmp_path/'initialization.json').read_text())['candidates']
    assert [r['status'] for r in rows] == ['rejected', 'failed', 'evaluated']


def test_budget_too_small_rejected_before_fitting(tmp_path):
    h = Harness()
    with pytest.raises(ValueError, match='increase --budget'):
        Search(h, Agents(), tmp_path, budget=2).run(candidates())
    assert not h.calls


def test_every_dataset_has_source_backed_valid_distinct_candidates():
    assert set(registry()['datasets']) == set(LOSO_DATASETS)
    for dataset in LOSO_DATASETS:
        pool = dataset_candidates(dataset)
        assert len(pool) == 4
        assert len({json.dumps(item['pipeline'], sort_keys=True) for item in pool}) == 4
        for item in pool:
            validate_pipeline(Pipeline.from_dict(item['pipeline']), {dataset})
            assert item['evidence']['sources'] and item['evidence']['adaptations']
        assert len({item['pipeline']['model']['architecture'] for item in pool}) == 2
        assert any(item['pipeline']['augmentation']['kind'] != 'none' for item in pool)
    assert dataset_candidates('unimib_shar')[0]['pipeline']['data']['window_seconds'] == 3.02
    assert dataset_candidates('hang_time')[0]['pipeline']['model']['lstm_layers'] == 1
    assert dataset_candidates('capture_24')[0]['pipeline']['data']['normalization'] == 'none'


def test_lodo_pool_uses_only_source_identity_and_no_target_records():
    class Task:
        protocol = 'lodo'
        train = [SimpleNamespace(dataset='uci_har'), SimpleNamespace(dataset='motion_sense')]
        @property
        def target(self):
            pytest.fail('LODO selection must not depend on target identity')
        @property
        def test(self):
            pytest.fail('LODO selection must not access held-out records')
    pool = initial_candidates(Task())
    assert pool[-1]['id'] == 'lodo/mobilehart/reference'
    assert all(p['id'].startswith(('uci_har/', 'motion_sense/', 'lodo/')) for p in pool)
    with pytest.raises(ValueError, match='supply --baseline'):
        dataset_candidates('unknown')


def test_minmax_statistics_fit_training_only_and_do_not_clip_validation():
    x = np.array([[[1., 4.], [3., 4.]]], dtype=np.float32)
    n = Normalizer('minmax').fit(x)
    np.testing.assert_allclose(n.transform(x), [[[0, 0], [1, 0]]])
    np.testing.assert_allclose(n.transform(np.array([[[5., 4.]]])), [[[2, 0]]])
    np.testing.assert_array_equal(n.mean, [[[1, 4]]])


@pytest.mark.parametrize('device', ['cpu'] + (['cuda'] if torch.cuda.is_available() else []))
def test_cli_dataset_selection_is_source_only_and_records_choice(manifest, tmp_path, monkeypatch, device):
    import sys
    from autohar_researcher import main, baselines
    from autohar_researcher.datasets import make_task
    task = make_task(manifest, 'loso', 'uci_har', '0')
    # Keep real ported models but shorten only the fixture's training horizon.
    pool = dataset_candidates('uci_har')
    for item in pool:
        item['pipeline']['model']['max_epochs'] = 1
        item['pipeline']['data']['window_seconds'] = 1
    monkeypatch.setattr(main, 'initial_candidates', lambda task: copy.deepcopy(pool))
    original = np.load
    protected = {r.path for r in task.test}
    def source_only(path, *args, **kwargs):
        assert path not in protected
        return original(path, *args, **kwargs)
    monkeypatch.setattr(np, 'load', source_only)
    output = tmp_path/'selected'
    monkeypatch.setattr(sys, 'argv', ['autohar', 'search', '--manifest', str(manifest),
        '--protocol', 'loso', '--target', 'uci_har', '--subject', '0', '--seeds', '13',
        '--budget', '4', '--rounds', '1', '--model', 'fixture-no-calls', '--device', device,
        '--output', str(output)])
    main.main()
    run = output/'seed-13/subject-0'
    report = json.loads((run/'initialization.json').read_text())
    metadata = json.loads((run/'run.json').read_text())
    assert metadata['baseline'] == report['selected_pipeline']
    assert metadata['initialization_selected_id'] == report['selected_id']
    assert report['evaluations'] == 4
    assert not (run/'final-test.started').exists()
