from types import SimpleNamespace

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


@pytest.mark.parametrize('judgment', [Judgment(True, 5, 5), Judgment(2, 2, 5), Judgment(2, 3, 3), Judgment(6, 5, 5)])
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
