import hashlib
import json
from pathlib import Path

import pytest
import torch

from autohar_researcher.agents import Search
from autohar_researcher.datasets import make_task
from autohar_researcher.harness import HARHarness
from autohar_researcher.planner import LLMAgents
from autohar_researcher.types import Pipeline


@pytest.mark.parametrize('name', ['ensemble', 'cascade'])
def test_composite_model_examples_train_all_members(name, manifest, tmp_path):
    torch.set_num_threads(1)
    path = Path(__file__).resolve().parents[1] / 'examples/models' / (name + '.json')
    p = Pipeline.from_dict(json.loads(path.read_text()))
    p.data['window_seconds'] = 1
    p.model['max_epochs'] = 1
    h = HARHarness(make_task(manifest, 'loso', 'uci_har', '0'), 13, tmp_path / name, 'cpu')
    h.preflight(p)
    model = h._build_model(p)
    logits = model(torch.randn(4, 50, 6))
    assert logits.shape == (4, 5)
    torch.nn.functional.cross_entropy(logits, torch.arange(4)).backward()
    assert all(parameter.grad is not None for parameter in model.parameters())
    result = h.evaluate(p)
    assert 0 <= result.score <= 1


def test_source_feedback_and_validation_probe_do_not_change_scoring(manifest, tmp_path, monkeypatch):
    torch.set_num_threads(1)
    p = Pipeline.from_dict({'data': {'window_seconds': 1},
        'model': {'architecture': 'cnn', 'hidden_dim': 4, 'depth': 1, 'max_epochs': 1},
        'augmentation': {'kind': 'jitter', 'strength': .1}})
    task = make_task(manifest, 'loso', 'uci_har', '0')
    h = HARHarness(task, 13, tmp_path / 'with-probes', 'cpu')
    result = h.evaluate(p)
    d = h._validation_diagnostics[h._key(p)]
    assert sum(d['real_training']['class_counts'].values()) == d['real_training']['window_count']
    assert len(d['real_validation']['channels']['mean']) == 6
    for name in ('training', 'validation'):
        probe = d[name + '_augmentation_probe']
        assert probe['diagnostic_only'] and not probe['used_for_selection']
        assert probe['before']['channels']['std'] != probe['after_conventional']['channels']['std']
        assert len(probe['conventional_predictions']['confusion_true_rows_predicted_columns']) == 5
    clean = HARHarness(task, 13, tmp_path / 'without-probes', 'cpu')
    monkeypatch.setattr(clean, '_augmentation_diagnostics', lambda *a: {})
    assert clean.evaluate(p).score == result.score
    key = hashlib.sha256(h._key(p).encode()).hexdigest()
    assert (h.output / 'diagnostics' / (key + '.json')).is_file()
    agents = LLMAgents(lambda *args: {})
    search = Search(h, agents, tmp_path / 'search')
    context = search._context(p, result, 1)
    assert context['evaluation_diagnostics_index'][0]['pipeline_id'] == key
    feedback = agents.inspect_evaluation({'pipeline_id': key, 'sections': ['real_validation']})
    assert feedback == {'real_validation': d['real_validation']}
    with pytest.raises(ValueError, match='Unknown diagnostic sections'):
        agents.inspect_evaluation({'pipeline_id': key, 'sections': ['test']})


@pytest.mark.parametrize('role', ['data', 'model', 'augmentation', 'judge', 'orchestrator'])
def test_every_role_can_request_evaluation_feedback(role):
    prompts = []
    def backend(called_role, prompt):
        assert called_role == role
        prompts.append(prompt)
        return {'inspect': {'pipeline_id': 'candidate'}} if len(prompts) == 1 else {'finished': True}
    agents = LLMAgents(backend)
    agents.inspect_evaluation = lambda query: {'confusion_true_rows_predicted_columns': [[3, 1], [0, 4]]}
    assert agents._call(role, 'Inspect source evidence') == {'finished': True}
    assert '[[3, 1], [0, 4]]' in prompts[1]


def test_diagnostic_inspections_stay_below_backend_input_limit():
    prompts = []
    def backend(role, prompt):
        assert len(prompt) < 750_000
        prompts.append(prompt)
        return {'inspect': {'pipeline_id': str(len(prompts))}} if len(prompts) < 5 else {'finished': True}
    agents = LLMAgents(backend)
    agents.inspect_evaluation = lambda query: {'id': query['pipeline_id'], 'details': 'x' * 350_000}
    assert agents._call('orchestrator', 'BASELINE CONTEXT') == {'finished': True}
    assert 'BASELINE CONTEXT' in prompts[-1]
    assert '"id": "4"' in prompts[-1]
    assert 'Earlier diagnostic responses summarized' in prompts[-1]


def test_oversized_fold_feedback_requests_narrower_inspection():
    prompts = []
    def backend(role, prompt):
        prompts.append(prompt)
        assert len(prompt) < 750_000
        return {'inspect': {'pipeline_id': 'candidate'}} if len(prompts) == 1 else {'finished': True}
    agents = LLMAgents(backend)
    agents.inspect_evaluation = lambda query: {'mean_validation_macro_f1': .8, 'folds': ['x' * 900_000]}
    assert agents._call('orchestrator', 'Source evidence') == {'finished': True}
    assert 'mean_validation_macro_f1' in prompts[-1]
    assert 'individual subject and sections' in prompts[-1]


def test_repeated_summary_preserves_class_evidence():
    from autohar_researcher.context_summary import diagnostic_summary, inspection_summary
    summary = diagnostic_summary({'validation_macro_f1': .7,
        'per_class': {'walking': {'f1': .4, 'support': 12}},
        'top_confusions': [{'true': 'walking', 'predicted': 'standing', 'count': 3}]})
    repeated = inspection_summary(inspection_summary(summary))
    assert repeated['weakest_classes'] == summary['weakest_classes']
    assert repeated['top_confusions'] == summary['top_confusions']
    assert repeated['validation_macro_f1'] == .7


def test_long_search_context_is_summarized_and_full_details_are_pageable(manifest, tmp_path):
    from autohar_researcher.types import Evaluation, Outcome
    p = Pipeline()
    h = HARHarness(make_task(manifest, 'loso', 'uci_har', '0'), 13, tmp_path / 'harness', 'cpu')
    detail = {'validation_macro_f1': .72, 'per_class': {'walking': {'f1': .4, 'support': 10}},
              'training_model_inputs': {'blob': 'x' * 80_000}}
    raw = {'mean_validation_macro_f1': .72,
           'folds': [{'subject': str(i), 'diagnostics': detail} for i in range(8)]}
    for i in range(60):
        candidate = Pipeline.from_dict({'model': {'hidden_dim': 64 + i}})
        h._validation_diagnostics[h._key(candidate)] = raw
    h._validation_diagnostics[h._key(p)] = raw
    search = Search(h, LLMAgents(lambda *args: {}), tmp_path / 'search', exploration=True)
    old = p.payload()
    old['code'] = {'research/components/model.py': '# ' + 'old source ' * 10_000}
    search.history = [Outcome(i, ('model',), 'evaluated', old, Evaluation(.6, {}, 1)) for i in range(200)]
    search.previous_search_history = [vars(row) | {'evaluation': vars(row.evaluation)} for row in search.history]
    search.experiment_ledger = [{'round': i, 'role': 'model', 'family': 'architecture',
        'changes': {'hidden_dim': i}, 'code_diff': 'diff ' * 30_000, 'rationale': 'test',
        'baseline_context': old, 'status': 'evaluated', 'score': .6, 'baseline_score': .7}
        for i in range(200)]
    context = search._context(p, Evaluation(.72, {}, 1), 201)
    assert len(json.dumps(context)) < 150_000
    assert context['history_summary']['total_outcomes'] == 200
    assert len(context['exploration']['experiment_ledger']) == 12
    assert context['source_validation_diagnostics']['mean_validation_macro_f1'] == .72
    key = context['baseline_pipeline_id']
    summary = search._inspect_evaluation({'pipeline_id': key})
    assert summary['summary_only'] and 'training_model_inputs' not in summary
    catalog = search._inspect_evaluation({'list': True, 'offset': 20})
    assert catalog['total'] == 60 and len(catalog['entries']) == 20
    parts, offset = [], 0
    while offset is not None:
        page = search._inspect_evaluation({'pipeline_id': key, 'subject': '0',
            'sections': ['training_model_inputs'], 'offset': offset})
        assert len(page['detail_page']) <= 32000
        parts.append(page['detail_page'])
        offset = page['next_offset']
    assert json.loads(''.join(parts)) == {'training_model_inputs': detail['training_model_inputs']}
    assert raw['folds'][0]['diagnostics']['training_model_inputs']['blob'] == 'x' * 80_000
