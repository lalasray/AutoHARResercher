from types import SimpleNamespace
import json

import pytest

from autohar_researcher.agents import Search
from autohar_researcher.patches import check_proposal, compose
from autohar_researcher.types import Evaluation, Judgment, Pipeline, Proposal


class Harness:
    sources = {'uci_har'}
    task = SimpleNamespace(public_metadata=lambda: {}, fingerprint='fixture')

    def __init__(self, scores=None):
        self.calls = []
        self.scores = iter(scores or [.5] * 50)

    def preflight(self, p):
        pass

    def evaluate(self, p):
        self.calls.append(p)
        return Evaluation(next(self.scores), {'uci_har': .5}, 1)


class Agents:
    def __init__(self, judgment=Judgment(3, 3, 4), failure=False):
        self.judgment, self.failure = judgment, failure

    def plan(self, context):
        return {'composition_bias': 'all'}

    def propose(self, role, context, plan):
        changes = {'data': {'overlap': .25}, 'model': {'hidden_dim': 8},
                   'augmentation': {'kind': 'jitter'}}[role]
        return Proposal(role, changes, {}, 'test')

    def judge(self, proposal, context, plan, size):
        if self.failure:
            raise ValueError('malformed LLM response')
        return self.judgment

    def compose(self, proposals, context, plan):
        return [['data', 'model'], ['data', 'model', 'augmentation']]


def test_judge_rejection_prevents_training(tmp_path):
    harness = Harness()
    search = Search(harness, Agents(Judgment(1, 5, 5)), tmp_path, rounds=1)
    search.run(Pipeline())
    assert len(harness.calls) == 1
    assert sum(o.status == 'rejected' for o in search.history) == 3


def test_malformed_judge_never_falls_back_to_approval(tmp_path):
    harness = Harness()
    Search(harness, Agents(failure=True), tmp_path, rounds=1).run(Pipeline())
    assert len(harness.calls) == 1


def test_budget_and_strict_promotion(tmp_path):
    harness = Harness([.5, .5, .7])
    search = Search(harness, Agents(), tmp_path, rounds=3, budget=3)
    selected, score = search.run(Pipeline())
    assert len(harness.calls) == 3
    assert score.score == .7
    assert selected.model['hidden_dim'] == 8
    assert selected.data['overlap'] == .5  # tie cannot promote data patch
    assert sum(o.status == 'promoted' for o in search.history) == 1


def test_ties_preserve_baseline(tmp_path):
    harness = Harness()
    search = Search(harness, Agents(), tmp_path, rounds=1)
    selected, score = search.run(Pipeline())
    assert selected.payload() == Pipeline().payload()
    assert not any(o.status == 'promoted' for o in search.history)
    assert len(harness.calls) == 6
    assert sum(o.status == 'evaluated' for o in search.history) == 5


def test_failed_training_counts_but_invalid_patch_does_not(tmp_path):
    harness = Harness([.5])
    search = Search(harness, Agents(), tmp_path, rounds=1, budget=2)
    search.run(Pipeline())
    assert search.evaluated == 2
    assert any(o.status == 'failed' for o in search.history)


@pytest.mark.parametrize('judgment', [Judgment(True, 5, 5), Judgment(2, 1, 5), Judgment(2, 3, 3), Judgment(6, 5, 5)])
def test_judge_thresholds(judgment):
    assert not judgment.eligible


def test_mutation_limits_and_protected_files():
    base = Pipeline()
    with pytest.raises(ValueError, match='protected'):
        check_proposal(base, Proposal('data', {}, {'autohar_researcher/harness.py': 'x = 1'}, ''), {'uci_har'})
    with pytest.raises(ValueError, match='Mutation exceeds'):
        check_proposal(base, Proposal('model', {}, {'research/components/model.py': '\n'.join(f'x{i}=1' for i in range(121))}, ''), {'uci_har'})
    with pytest.raises(ValueError, match='protected'):
        check_proposal(base, Proposal('data', {'resample_hz': 100}, {}, ''), {'uci_har'})


def test_dependencies_and_overlap():
    patch = Proposal('model', {'hidden_dim': 8}, {}, '', ('data',))
    with pytest.raises(ValueError, match='dependencies'):
        compose(Pipeline(), [patch])


def test_shape_import_and_field_limits():
    with pytest.raises(ValueError, match='Unapproved'):
        check_proposal(Pipeline(), Proposal('data', {}, {'research/components/data.py': 'import os'}, ''), {'uci_har'})
    with pytest.raises(ValueError, match='Mutation exceeds'):
        check_proposal(Pipeline(), Proposal('model', {str(i): i for i in range(9)}, {}, ''), {'uci_har'})


def test_budget_bounds(tmp_path):
    with pytest.raises(ValueError):
        Search(Harness(), Agents(), tmp_path, rounds=101)


def test_single_agent_comparator_uses_same_harness_without_judge(tmp_path):
    from autohar_researcher.agents import SingleAgentSearch
    class Generalist:
        def generalist(self, context):
            width = context['baseline']['model']['hidden_dim']
            return [Proposal('model', {'hidden_dim': width + 1}, {}, 'test')]
    harness = Harness([.5, .6, .7])
    search = SingleAgentSearch(harness, Generalist(), tmp_path, rounds=1, budget=3)
    selected, score = search.run(Pipeline())
    assert score.score == .7
    assert selected.model['hidden_dim'] == Pipeline().model['hidden_dim'] + 2
    assert len(harness.calls) == 3
    assert search.improvements == 2
    assert search.stop_reason == 'fit_budget_exhausted'
    saved = json.loads((tmp_path / 'selected.json').read_text())
    assert saved['improvements'] == 2
    assert saved['stop_reason'] == 'fit_budget_exhausted'
    assert saved['evaluations'] == 3
    assert saved['validation']['score'] == .7


def test_exploratory_judge_accepts_coherent_uncertain_hypothesis():
    assert Judgment(2, 2, 4).eligible
    assert not Judgment(2, 2, 3).eligible


def test_stops_after_target_number_of_improving_rounds(tmp_path):
    harness = Harness([.5, .6, .7, .65, .64, .63])
    search = Search(harness, Agents(), tmp_path, rounds=10, budget=30, target_improvements=1)
    _, score = search.run(Pipeline())
    assert score.score == .7
    assert search.improvements == 1
    assert search.stop_reason == 'target_improvements_reached'
    assert max(o.round for o in search.history) == 1


def test_perfect_validation_stops_unattainable_improvement_target(tmp_path):
    harness = Harness([1.0])
    search = Search(harness, Agents(), tmp_path, target_improvements=10)
    search.run(Pipeline())
    assert len(harness.calls) == 1
    assert search.improvements == 0
    assert search.stop_reason == 'perfect_validation_score_no_further_improvement_possible'


class CountingGeneralist:
    def __init__(self, reject=False):
        self.calls = 0
        self.reject = reject

    def generalist(self, context):
        self.calls += 1
        if self.reject:
            return []
        width = context['baseline']['model']['hidden_dim'] + self.calls
        return [Proposal('model', {'hidden_dim': width}, {}, 'fixture')]


def assert_single_agent_metadata(search, path):
    saved = json.loads((path / 'selected.json').read_text())
    promotions = sum(row.status == 'promoted' for row in search.history)
    assert saved['improvements'] == search.improvements == promotions
    assert saved['evaluations'] == search.evaluated
    assert saved['stop_reason'] == search.stop_reason
    assert saved['stop_reason'] is not None
    disk_history = [json.loads(row) for row in (path / 'history.jsonl').read_text().splitlines()]
    assert sum(row['status'] == 'promoted' for row in disk_history) == promotions
    return saved


@pytest.mark.parametrize('values', [[.5, .4, .3, .2], [.5]*4, [.5, float('nan'), float('inf'), .4]])
def test_single_agent_unsuccessful_fits_consume_budget_without_promotions(tmp_path, values):
    from autohar_researcher.agents import SingleAgentSearch
    h, agent = Harness(values), CountingGeneralist()
    search = SingleAgentSearch(h, agent, tmp_path, rounds=5, budget=4)
    _, result = search.run(Pipeline())
    assert result.score == .5
    assert len(h.calls) == 4 and agent.calls == 3
    assert search.improvements == 0 and search.stop_reason == 'fit_budget_exhausted'
    saved = assert_single_agent_metadata(search, tmp_path)
    assert saved['validation']['score'] == .5


def test_single_agent_failed_fits_consume_budget_without_promotions(tmp_path):
    from autohar_researcher.agents import SingleAgentSearch
    class FailingHarness(Harness):
        def evaluate(self, p):
            if self.calls:
                self.calls.append(p)
                raise RuntimeError('fit failed')
            return super().evaluate(p)
    h = FailingHarness([.5])
    search = SingleAgentSearch(h, CountingGeneralist(), tmp_path, rounds=5, budget=3)
    search.run(Pipeline())
    assert len(h.calls) == 3
    assert sum(row.status == 'failed' for row in search.history) == 2
    assert search.stop_reason == 'fit_budget_exhausted'
    assert_single_agent_metadata(search, tmp_path)


def test_single_agent_rejections_do_not_consume_fit_budget(tmp_path):
    from autohar_researcher.agents import SingleAgentSearch
    h, agent = Harness([.5]), CountingGeneralist(reject=True)
    search = SingleAgentSearch(h, agent, tmp_path, rounds=2, budget=20)
    search.run(Pipeline())
    assert agent.calls == 6 and len(h.calls) == 1
    assert sum(row.status == 'rejected' for row in search.history) == 6
    assert search.stop_reason == 'round_limit_reached'
    assert_single_agent_metadata(search, tmp_path)


def test_single_agent_round_limit_preserves_three_attempts_per_round(tmp_path):
    from autohar_researcher.agents import SingleAgentSearch
    h, agent = Harness([.5, .6, .7, .8, .9, .95, .99]), CountingGeneralist()
    search = SingleAgentSearch(h, agent, tmp_path, rounds=2, budget=20)
    _, result = search.run(Pipeline())
    assert agent.calls == 6 and search.evaluated == 7 and search.improvements == 6
    assert result.score == .99 and search.stop_reason == 'round_limit_reached'
    assert [row.round for row in search.history if row.status == 'promoted'] == [1,1,1,2,2,2]
    assert assert_single_agent_metadata(search, tmp_path)['round'] == 2


def test_single_agent_baseline_exhausts_budget_before_proposals(tmp_path):
    from autohar_researcher.agents import SingleAgentSearch
    h, agent = Harness([.5]), CountingGeneralist()
    search = SingleAgentSearch(h, agent, tmp_path, rounds=5, budget=1)
    search.run(Pipeline())
    assert len(h.calls) == 1 and agent.calls == 0
    assert search.stop_reason == 'fit_budget_exhausted'
    assert assert_single_agent_metadata(search, tmp_path)['round'] == 0


@pytest.mark.parametrize('budget', [2, 20])
def test_single_agent_promotion_target_is_counted_before_next_attempt(tmp_path, budget):
    from autohar_researcher.agents import SingleAgentSearch
    h, agent = Harness([.5, .7]), CountingGeneralist()
    search = SingleAgentSearch(h, agent, tmp_path, rounds=5, budget=budget, target_improvements=1)
    search.run(Pipeline())
    assert agent.calls == 1 and len(h.calls) == 2 and search.improvements == 1
    assert search.stop_reason == 'target_improvements_reached'
    assert_single_agent_metadata(search, tmp_path)


def test_single_agent_perfect_baseline_reports_unattainable_target(tmp_path):
    from autohar_researcher.agents import SingleAgentSearch
    h, agent = Harness([1.]), CountingGeneralist()
    search = SingleAgentSearch(h, agent, tmp_path, rounds=5, budget=20, target_improvements=2)
    search.run(Pipeline())
    assert len(h.calls) == 1 and agent.calls == 0
    assert search.stop_reason == 'perfect_validation_score_no_further_improvement_possible'
    assert_single_agent_metadata(search, tmp_path)


class BrokenComposer(Agents):
    def __init__(self, response):
        super().__init__()
        self.response = response
        self.composer_calls = 0

    def compose(self, proposals, context, plan):
        self.composer_calls += 1
        if isinstance(self.response, Exception):
            raise self.response
        return self.response


@pytest.mark.parametrize('response', [RuntimeError('backend unavailable'), None, {}, {'subsets': []}, 'data', 7])
def test_composer_failure_evaluates_only_eligible_singletons_and_persists(tmp_path, response):
    harness = Harness([.5, .6, .7, .55])
    search = Search(harness, BrokenComposer(response), tmp_path, rounds=1, budget=4)
    selected, score = search.run(Pipeline())
    assert score.score == .7 and selected.model['hidden_dim'] == 8
    assert len(harness.calls) == search.evaluated == 4
    for row in search.history:
        if row.status == 'evaluated':
            assert len(row.origin) == 1 and row.origin[0] != 'orchestrator'
    failures = [row for row in search.history if row.origin == ('orchestrator',) and row.status == 'failed']
    assert len(failures) == 1 and 'Composer failed' in failures[0].reason
    saved = json.loads((tmp_path/'selected.json').read_text())
    assert saved['validation']['score'] == .7
    assert saved['evaluations'] == 4 and saved['improvements'] == 1
    assert saved['stop_reason'] == 'fit_budget_exhausted'
    artifact = json.loads((tmp_path/'round-001.json').read_text())
    assert artifact['composition']['status'] == 'failed'
    assert artifact['subsets'] == [['data'], ['model'], ['augmentation']]
    timing = json.loads((tmp_path/'round-timings.jsonl').read_text().splitlines()[0])
    assert timing['candidate_evaluations'] == 3


@pytest.mark.parametrize('response', [{}, {'subsets': None}, {'subsets': 'data'}, [], None])
def test_llm_missing_or_malformed_subsets_uses_same_recovery(tmp_path, monkeypatch, response):
    from autohar_researcher.planner import LLMAgents
    composer = LLMAgents(None)
    monkeypatch.setattr(composer, '_call', lambda *args: response)
    agents = Agents()
    agents.compose = composer.compose
    harness = Harness([.5, .6, .7, .55])
    search = Search(harness, agents, tmp_path, rounds=1, budget=4)
    _, score = search.run(Pipeline())
    assert score.score == .7 and search.evaluated == 4
    failure = next(row for row in search.history if row.origin == ('orchestrator',))
    assert failure.status == 'failed' and 'subsets list' in failure.reason
    assert (tmp_path/'selected.json').is_file()


@pytest.mark.parametrize('subset', [None, [], 'data', {}, [['data']], [{'role': 'data'}],
                                   ['data', 'data'], ['unknown'], [1], ['model', None]])
def test_invalid_combination_entries_never_train_or_crash(tmp_path, subset):
    search = Search(Harness(), BrokenComposer([subset]), tmp_path, rounds=1, budget=10)
    search.run(Pipeline())
    assert search.evaluated == 4  # Baseline plus three eligible singletons only.
    assert all(len(row.origin) == 1 for row in search.history if row.status == 'evaluated')
    assert any(row.origin == ('orchestrator',) and row.status == 'rejected' for row in search.history)
    artifact = json.loads((tmp_path/'round-001.json').read_text())
    assert artifact['composition']['rejected_subset_indices'] == [0]
    assert artifact['composition']['status'] == 'invalid_subsets_rejected'


def test_valid_combinations_still_run_when_other_entries_are_malformed(tmp_path):
    search = Search(Harness(), BrokenComposer([{'bad': []}, ['model','data'], ['data','model']]),
                    tmp_path, rounds=1, budget=10)
    search.run(Pipeline())
    assert search.evaluated == 5  # One valid, deduplicated combination.
    origins = [row.origin for row in search.history if row.status == 'evaluated']
    assert origins.count(('data','model')) == 1


@pytest.mark.parametrize('budget', [1, 2, 3, 4])
def test_composer_failure_never_consumes_or_exceeds_fit_budget(tmp_path, budget):
    harness, agents = Harness(), BrokenComposer(RuntimeError('backend failure'))
    search = Search(harness, agents, tmp_path, rounds=10, budget=budget)
    search.run(Pipeline())
    assert search.evaluated == len(harness.calls) == budget
    assert search.stop_reason == 'fit_budget_exhausted'
    assert agents.composer_calls == (0 if budget == 1 else 1)
    saved = json.loads((tmp_path/'selected.json').read_text())
    assert saved['evaluations'] == budget


def test_composer_cannot_mutate_singletons_before_failing(tmp_path):
    class MutatingComposer(Agents):
        def compose(self, proposals, context, plan):
            proposals['model'].changes['hidden_dim'] = -999
            proposals.clear()
            raise RuntimeError('failed after modifying input')
    harness = Harness()
    search = Search(harness, MutatingComposer(), tmp_path, rounds=1, budget=4)
    search.run(Pipeline())
    assert search.evaluated == 4
    assert harness.calls[2].model['hidden_dim'] == 8


def test_composer_failure_does_not_bypass_singleton_dependencies(tmp_path):
    class DependentAgent(BrokenComposer):
        def propose(self, role, context, plan):
            proposal = super().propose(role, context, plan)
            if role == 'model':
                return Proposal(proposal.role, proposal.changes, proposal.files, proposal.rationale,
                                ('data',), proposal.validation)
            return proposal
    harness = Harness()
    search = Search(harness, DependentAgent(RuntimeError('offline')), tmp_path, rounds=1, budget=10)
    search.run(Pipeline())
    assert search.evaluated == 3
    assert any(row.origin == ('model',) and row.reason == 'Unsatisfied proposal dependencies'
               for row in search.history if row.status == 'rejected')
    assert all(row.model['hidden_dim'] == Pipeline().model['hidden_dim'] for row in harness.calls)


def test_combinations_still_require_preflight_and_do_not_charge_invalid_fits(tmp_path):
    class RejectCombined(Harness):
        def preflight(self, pipeline):
            if pipeline.data['overlap'] == .25 and pipeline.model['hidden_dim'] == 8:
                raise ValueError('conflicting combined interfaces')
    harness = RejectCombined()
    search = Search(harness, BrokenComposer([['data','model']]), tmp_path, rounds=1, budget=10)
    search.run(Pipeline())
    assert search.evaluated == len(harness.calls) == 4
    assert any(row.origin == ('data','model') and row.status == 'rejected' for row in search.history)
