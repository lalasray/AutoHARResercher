import json

import pytest

from autohar_researcher.exploration import FAMILIES, LIMITS, entry, summarize, validate_plan
from autohar_researcher.patches import check_proposal, compose
from autohar_researcher.types import Pipeline, Proposal
from autohar_researcher.agents import Search


def test_coverage_directs_stagnation_to_untried_families():
    ledger = [{'role': role, 'family': families[-1], 'code_diff': '', 'changes': {}} for role, families in FAMILIES.items()]
    context = summarize(ledger, 8)
    for role, families in FAMILIES.items():
        assert families[-1] not in context['allowed_focus'][role]
        assert families[0] in context['allowed_focus'][role]
    assert context['mutation_limits'] == LIMITS


def test_plan_requires_reasoned_allowed_focus_for_all_specialists():
    context = summarize([], 0)
    plan = {'hypotheses': {role: dict(family=families[0], hypothesis='A mechanism', evidence='Untested', expected_effect='Reduce variance', falsification='Validation regression') for role, families in FAMILIES.items()}}
    validate_plan(plan, context)
    plan['hypotheses']['data']['family'] = 'unknown'
    with pytest.raises(ValueError, match='allowed exploration family'):
        validate_plan(plan, context)


def test_expanded_edits_and_override_removal():
    base = Pipeline()
    source = '\n'.join(f'def helper_{i}(x):\n    return x' for i in range(5))
    proposal = Proposal('model', {}, {'research/components/model.py': source}, 'larger experiment')
    with pytest.raises(ValueError, match='Mutation exceeds'):
        check_proposal(base, proposal, {'uci_har'})
    assert check_proposal(base, proposal, {'uci_har'}, LIMITS)['functions'] == 5
    base.code['research/components/data.py'] = 'def preprocess(x):\n    return x * .5'
    remove = Proposal('data', {'window_seconds': .5, 'overlap': .9}, {'research/components/data.py': None}, 'Windowing ablation')
    check_proposal(base, remove, {'uci_har'}, LIMITS)
    assert 'research/components/data.py' not in compose(base, [remove]).code


def test_architecture_switch_clears_previous_custom_model():
    base = Pipeline()
    base.code['research/components/model.py'] = 'def build_model(a,b,c):\n    return None'
    p = compose(base, [Proposal('model', {'architecture':'tinyhar'}, {}, 'switch')])
    assert not p.code


def test_duplicate_detection_ignores_comments_and_formatting():
    a, b = Pipeline(), Pipeline()
    a.code['research/components/data.py'] = 'def preprocess(x):\n    return x * .5'
    b.code['research/components/data.py'] = '# same operation\ndef preprocess(x):\n  return x * 0.5\n'
    assert Search._key(a) == Search._key(b)


def test_ledger_retains_concrete_changes_and_code_diff():
    p = Proposal('data', {'window_seconds': 3}, {'research/components/data.py':'def preprocess(x):\n    return x.clip(-10,10)'}, 'Temporal context hypothesis')
    row = entry(1, 'data', p, Pipeline())
    assert row['family'] == 'window_stride'
    compact = summarize([row], 0)['experiment_ledger'][0]
    assert compact['changes'] == {'window_seconds': 3}
    assert 'clip(-10,10)' in compact['code_diff']


def test_directed_planner_repairs_invalid_family_without_training():
    from autohar_researcher.planner import LLMAgents
    context = {'exploration': summarize([], 0)}
    good = {'priority': list(FAMILIES), 'hypotheses': {role: dict(family=families[0], hypothesis='Mechanism', evidence='Not yet tested', expected_effect='Change recall', falsification='No gain') for role, families in FAMILIES.items()}}
    responses = iter([{'hypotheses': {}}, good])
    calls = []
    def backend(role, prompt):
        calls.append(prompt)
        return next(responses)
    assert LLMAgents(backend).plan(context) == good
    assert len(calls) == 2
    assert 'Correct this plan schema error' in calls[-1]


def test_failure_memory_separates_evidence_from_untrained_rejections():
    from autohar_researcher.exploration import failure_memory
    common=dict(role='model',round=1,baseline_score=.7)
    rows=[dict(common,status='evaluated',score=.6),dict(common,status='evaluated',score=.8),
          dict(common,status='judge_rejected',score=None),dict(common,status='failed',score=None)]
    result=failure_memory(rows)
    assert len(result)==2
    assert [r['status'] for r in result]==['evaluated','failed']


def test_repeat_review_blocks_training_and_records_reason(tmp_path):
    from test_search import Harness, Agents
    class Repeats(Agents):
        def review_repeat(self,proposal,context):
            return dict(repeat=True,matched_round=1,reason='Same failed mechanism with renamed variables')
    harness=Harness();search=Search(harness,Repeats(),tmp_path,exploration=True)
    context={'round':2,'exploration':{'failed_ideas':[{'round':1,'role':'model'}]}}
    with pytest.raises(ValueError,match='Repeated failed idea'):
        search._proposal('model',Pipeline(),context,{})
    assert not harness.calls
    assert json.loads((tmp_path/'repeat-review-002-model.json').read_text())['repeat'] is True


def test_repeat_review_demands_real_reference_and_material_difference():
    from autohar_researcher.planner import LLMAgents
    context={'baseline':Pipeline().payload(),'exploration':{'failed_ideas':[{'round':2,'role':'model'}]}}
    proposal=Proposal('model',{'learning_rate':.0003},{},'A different learning rate')
    result=dict(repeat=False,matched_round=2,reason='Parameter differs',material_difference='Learning rate .001 to .0003')
    agents=LLMAgents(lambda role,prompt:result)
    assert agents.review_repeat(proposal,context)['repeat'] is False
    result.update(repeat=True,matched_round=999)
    with pytest.raises(ValueError,match='recorded failed trial'):
        agents.review_repeat(proposal,context)
