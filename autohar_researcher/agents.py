from __future__ import annotations

from research.modules.architectures.provenance import model_provenance
from research.augmentations import policy_provenance

from .context_summary import diagnostic_summary, history_summary, outcome_summary

import copy
import json
import math
import time
import hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path
from typing import Protocol

from .patches import check_proposal, compose
from .types import Evaluation, Judgment, Outcome, Pipeline, Proposal, ROLES


class Agents(Protocol):
    def plan(self, context: dict) -> dict: ...
    def propose(self, role: str, context: dict, plan: dict) -> Proposal: ...
    def judge(self, proposal: Proposal, context: dict, plan: dict, size: dict) -> Judgment: ...
    def compose(self, proposals: dict[str, Proposal], context: dict, plan: dict) -> list[list[str]]: ...


class Search:
    def __init__(self, harness, agents: Agents, output: Path, rounds: int = 100,
                 budget: int = 300, target_improvements: int | None = None, exploration: bool = False):
        if type(rounds) is not int or type(budget) is not int or rounds < 1 or budget < 1:
            raise ValueError('Rounds and empirical evaluation budget must be positive integers')
        if not exploration and (rounds > 100 or budget > 300):
            raise ValueError('Paper limits are 100 rounds and 300 empirical evaluations')
        if target_improvements is not None and (type(target_improvements) is not int or target_improvements < 1):
            raise ValueError('target_improvements must be a positive integer')
        self.exploration = exploration
        self.experiment_ledger = []
        self.stagnant_rounds = 0
        self.target_improvements = target_improvements
        self.improvements = 0
        self.stop_reason = None
        self.harness, self.agents, self.output = harness, agents, output
        self.rounds, self.budget = rounds, budget
        self.evaluated = 0
        self.history: list[Outcome] = []
        self.output.mkdir(parents=True, exist_ok=True)
        if (self.output / 'history.jsonl').exists():
            raise ValueError('Use a fresh output directory; resuming requires an audited checkpoint')
        self._seen = set()
        if hasattr(agents, 'inspect_evaluation'):
            agents.inspect_evaluation = self._inspect_evaluation

    def _record(self, outcome: Outcome):
        self.history.append(outcome)
        with (self.output / 'history.jsonl').open('a') as f:
            f.write(json.dumps(asdict(outcome), allow_nan=False) + '\n')

    @staticmethod
    def _key(p: Pipeline):
        import ast
        payload = p.payload()
        payload['code'] = {path: ast.dump(ast.parse(source), include_attributes=False)
                           for path, source in payload['code'].items()}
        return json.dumps(payload, sort_keys=True)

    def _inspect_evaluation(self, query):
        diagnostics = getattr(self.harness, '_validation_diagnostics', {})
        if query.get('list'):
            offset = query.get('offset', 0)
            if type(offset) is not int or offset < 0:
                raise ValueError('Catalog offset must be nonnegative')
            entries = list(reversed(list(diagnostics.items())))
            return {'total': len(entries), 'entries': [{'pipeline_id': hashlib.sha256(key.encode()).hexdigest(), 'validation_macro_f1': value.get('mean_validation_macro_f1', value.get('validation_macro_f1')), 'sections': list(value)} for key,value in entries[offset:offset+20]], 'next_offset': offset+20 if offset+20 < len(entries) else None}
        key = next((key for key in diagnostics
                    if hashlib.sha256(key.encode()).hexdigest() == query.get('pipeline_id')), None)
        if key is None:
            outcome = next((row for row in reversed(self.history) if row.pipeline and
                hashlib.sha256(json.dumps(row.pipeline, sort_keys=True).encode()).hexdigest() == query.get('pipeline_id')), None)
            if outcome is None:
                raise ValueError('Unknown evaluated pipeline ID')
            result = {'pipeline': outcome.pipeline, 'status': outcome.status, 'reason': outcome.reason,
                      'validation_macro_f1': outcome.evaluation.score if outcome.evaluation else None}
        else:
            result = {**diagnostics[key], 'pipeline': json.loads(key)}
        if query.get('subject') is not None:
            result = next((fold['diagnostics'] for fold in result.get('folds', [])
                           if fold['subject'] == str(query['subject'])), None)
            if result is None:
                raise ValueError('Unknown diagnostic fold')
        sections = query.get('sections')
        if sections is not None:
            if not isinstance(sections, list) or any(section not in result for section in sections):
                raise ValueError('Unknown diagnostic sections')
            result = {section: result[section] for section in sections}
        if sections is None:
            return diagnostic_summary(result)
        text = json.dumps(result)
        if len(text) > 32000 or 'offset' in query:
            offset = query.get('offset', 0)
            if type(offset) is not int or offset < 0:
                raise ValueError('Diagnostic page offset must be nonnegative')
            return {'detail_page': text[offset:offset+32000], 'offset': offset, 'total_characters': len(text), 'next_offset': offset+32000 if offset+32000 < len(text) else None, 'encoding': 'JSON text; request next_offset to continue'}
        return copy.deepcopy(result)

    def _diagnostic_context(self, baseline):
        diagnostics = getattr(self.harness, '_validation_diagnostics', {})
        baseline_key = (self.harness._key(baseline) if hasattr(self.harness, '_key')
                        else json.dumps(baseline.payload(), sort_keys=True))
        current = diagnostics.get(baseline_key, {})
        return {'source_validation_diagnostics': diagnostic_summary(current),
                'baseline_pipeline_id': hashlib.sha256(baseline_key.encode()).hexdigest(),
                'evaluation_diagnostics_index': [
                    {'pipeline_id': hashlib.sha256(key.encode()).hexdigest(),
                     'sections': list(value) + ['pipeline'],
                     'architecture': json.loads(key)['model']['architecture'],
                     'validation_macro_f1': value.get('mean_validation_macro_f1', value.get('validation_macro_f1')),
                     'outcomes': [{'round': o.round, 'status': o.status, 'origin': o.origin}
                                  for o in self.history if o.pipeline and json.dumps(o.pipeline, sort_keys=True) == key][-5:],
                     'subjects': [fold['subject'] for fold in value.get('folds', [])],
                     'fold_sections': list(value.get('folds', [{}])[0].get('diagnostics', {}))}
                    for key, value in list(diagnostics.items())[-20:]],
                'diagnostic_catalog_total': len(diagnostics),
                'diagnostic_access': 'Return {"inspect": {"pipeline_id": "ID", "subject": "optional fold", "sections": ["section"]}} to inspect any evaluated run. Omit sections for a compact summary. Explicit sections return details, paginated with offset/next_offset when large. Use {"list":true,"offset":0} to page through older candidate IDs.'}

    def _context(self, baseline: Pipeline, score: Evaluation, round_id: int) -> dict:
        from research.augmentations import catalog
        from pathlib import Path
        name = baseline.model['architecture']
        source = Path(__file__).resolve().parents[1] / 'research/modules/architectures' / f'{name}.py'
        model_source = source.read_text() if name.isidentifier() and source.is_file() else ''
        from .exploration import summarize
        from research.modules.architectures.presets import PRESETS
        return {**self._diagnostic_context(baseline), 'architecture_presets': PRESETS, 'model_operation_source': (source.parent / 'torch_author.py').read_text() if self.exploration else '', 'exploration': summarize(self.experiment_ledger, self.stagnant_rounds) if self.exploration else None, 'previous_search_history': history_summary(getattr(self, 'previous_search_history', [])), 'model_source_context': model_source, 'starting_point_evidence': getattr(self, 'starting_point_evidence', {}), 'augmentation_catalog': catalog(), 'task': self.harness.task.public_metadata(), 'round': round_id,
                'baseline': baseline.payload(), 'baseline_validation': asdict(score),
                'remaining_evaluations': self.budget - self.evaluated,
                'history': [outcome_summary(asdict(o)) for o in self.history[-10:]],
                'history_summary': history_summary([asdict(o) for o in self.history]),
                'available_generators': sorted(self.harness.generator.servers) if getattr(self.harness, 'generator', None) else []}

    def _proposal(self, role: str, base: Pipeline, context: dict, plan: dict):
        proposal = self.agents.propose(role, copy.deepcopy(context), copy.deepcopy(plan))
        if proposal.role != role:
            raise ValueError('Specialist returned the wrong role')
        from .exploration import LIMITS
        size = check_proposal(base, proposal, self.harness.sources, limits=LIMITS)
        failed = context.get('exploration', {}).get('failed_ideas', []) if context.get('exploration') else []
        if failed and hasattr(self.agents, 'review_repeat'):
            review = self.agents.review_repeat(proposal, copy.deepcopy(context))
            (self.output / f'repeat-review-{context["round"]:03d}-{role}.json').write_text(json.dumps(review, indent=2))
            if review['repeat']:
                raise ValueError('Repeated failed idea: ' + review['reason'])
        judgment = self.agents.judge(proposal, copy.deepcopy(context), copy.deepcopy(plan), size)
        if not judgment.eligible:
            return proposal, judgment, size, False
        return proposal, judgment, size, True

    def _evaluate_batch(self, pipelines):
        workers = getattr(self.harness, 'candidate_workers', 1)
        if workers > 1:
            from .parallel import evaluate_many
            yield from evaluate_many(self.harness, pipelines, workers, self.harness.worker_threads)
        else:
            for pipeline in pipelines:
                try:
                    yield self.harness.evaluate(pipeline)
                except Exception as exc:
                    yield exc

    def _initialize(self, initial):
        explicit = isinstance(initial, Pipeline)
        items = ([{'id': 'explicit', 'pipeline': initial.payload(),
                   'evidence': {'selection': 'Explicit pipeline configuration'}}] if explicit else copy.deepcopy(initial))
        if not items:
            raise ValueError('Initial candidate pool is empty')
        if len(items) > self.budget:
            raise ValueError(f'Initial pool has {len(items)} candidates but budget is {self.budget}; increase --budget or supply --baseline')
        report = {'selection_metric': 'source_validation_macro_f1', 'tie_break': 'first candidate',
                  'task_fingerprint': self.harness.task.fingerprint, 'candidates': [], 'selected_id': None}
        best = None
        def persist():
            (self.output / 'initialization.json').write_text(json.dumps(report, indent=2, allow_nan=False))
        persist()
        pending = []
        for item in items:
            row = copy.deepcopy(item)
            report['candidates'].append(row)
            candidate = Pipeline.from_dict(item['pipeline'])
            key = self._key(candidate)
            if key in self._seen:
                row['status'] = 'duplicate'
                persist()
                continue
            try:
                self.harness.preflight(candidate)
            except Exception as exc:
                row.update(status='rejected', reason=str(exc))
                self._record(Outcome(0, ('initialization', item['id']), 'rejected', candidate.payload(), reason=str(exc)))
                persist()
                if explicit:
                    raise
                continue
            self._seen.add(key)
            self.evaluated += 1
            row['status'] = 'training'
            persist()
            pending.append((item, row, candidate))
        for (item, row, candidate), score in zip(pending, self._evaluate_batch([x[2] for x in pending])):
            try:
                if isinstance(score, Exception):
                    raise score
                if not math.isfinite(score.score):
                    raise ValueError('Non-finite baseline score')
            except Exception as exc:
                row.update(status='failed', reason=str(exc))
                self._record(Outcome(0, ('initialization', item['id']), 'failed', candidate.payload(), reason=str(exc)))
                persist()
                if explicit:
                    raise
                continue
            row.update(status='evaluated', validation=asdict(score))
            if not explicit:
                self._record(Outcome(0, ('initialization', item['id']), 'evaluated', candidate.payload(), score))
            if best is None or score.score > best[1].score:
                best = (copy.deepcopy(candidate), score)
                report['selected_id'] = item['id']
                report['selected_pipeline'] = candidate.payload()
                report['selected_validation'] = asdict(score)
                self.starting_point_evidence = copy.deepcopy(item.get('evidence', {}))
            report['evaluations'] = self.evaluated
            persist()
        if best is None:
            raise ValueError('No initial candidate succeeded; inspect initialization.json')
        self._record(Outcome(0, ('baseline',), 'baseline', best[0].payload(), best[1]))
        return best

    def _limit_stop_reason(self, score: Evaluation):
        if self.target_improvements is not None and self.improvements >= self.target_improvements:
            return 'target_improvements_reached'
        if self.target_improvements is not None and score.score >= 1.0:
            return 'perfect_validation_score_no_further_improvement_possible'
        if self.evaluated >= self.budget:
            return 'fit_budget_exhausted'
        return None

    def run(self, initial) -> tuple[Pipeline, Evaluation]:
        baseline, score = self._initialize(initial)
        for round_id in range(1, self.rounds + 1):
            round_started, evaluated_before = time.monotonic(), self.evaluated
            self.stop_reason = self._limit_stop_reason(score)
            if self.stop_reason is not None:
                break
            print(f'Round {round_id}: validation={score.score:.6f}, evaluations={self.evaluated}/{self.budget}', flush=True)
            context = self._context(baseline, score, round_id)
            plan = self.agents.plan(copy.deepcopy(context))
            if self.exploration:
                from .exploration import validate_plan
                validate_plan(plan, context['exploration'])
            eligible = {}
            reviews = {}
            with ThreadPoolExecutor(max_workers=3) as pool:
                futures = {pool.submit(self._proposal, role, baseline, context, plan): role for role in ROLES}
                for future in as_completed(futures):
                    role = futures[future]
                    try:
                        proposal, judgment, size, accepted = future.result()
                        reviews[role] = {'proposal': asdict(proposal), 'judgment': asdict(judgment), 'size': size}
                        if self.exploration:
                            from .exploration import entry
                            row = entry(round_id, role, proposal, baseline, plan['hypotheses'][role]['family'])
                            row['baseline_score'] = score.score
                            row['baseline_signature'] = hashlib.sha256(json.dumps(baseline.payload(),sort_keys=True).encode()).hexdigest()
                            row['status'] = 'eligible' if accepted else 'judge_rejected'
                            self.experiment_ledger.append(row)
                        if accepted:
                            eligible[role] = proposal
                        else:
                            self._record(Outcome(round_id, (role,), 'rejected', reason=judgment.rationale or 'Judge rubric threshold'))
                    except Exception as exc:
                        self._record(Outcome(round_id, (role,), 'rejected', reason=str(exc)))

            selected = []
            composition = {'status': 'skipped' if not eligible else 'completed', 'rejected_subset_indices': []}
            if eligible:
                try:
                    recommendations = self.agents.compose(copy.deepcopy(eligible), copy.deepcopy(context), copy.deepcopy(plan))
                    if not isinstance(recommendations, list):
                        raise ValueError('Composer recommendations must be a list of role subsets')
                    selected = recommendations
                except Exception as exc:
                    composition.update(status='failed', error_type=type(exc).__name__, reason=str(exc))
                    self._record(Outcome(round_id, ('orchestrator',), 'failed',
                                         reason=f'Composer failed ({type(exc).__name__}): {exc}'))
            priority = plan.get('priority', list(ROLES))
            if not isinstance(priority, list) or len(priority) != 3 or set(priority) != set(ROLES):
                raise ValueError('Orchestrator priority must be a permutation of specialist roles')
            subsets = [(role,) for role in priority if role in eligible]
            for index, subset in enumerate(selected):
                if (not isinstance(subset, list) or not subset or not all(isinstance(role, str) for role in subset)
                        or len(set(subset)) != len(subset) or not set(subset) <= eligible.keys()):
                    composition['rejected_subset_indices'].append(index)
                    self._record(Outcome(round_id, ('orchestrator',), 'rejected', reason='Invalid composition subset'))
                    continue
                key = tuple(role for role in ROLES if role in subset)
                if key not in subsets:
                    subsets.append(key)
            if composition['rejected_subset_indices']:
                composition['status'] = 'invalid_subsets_rejected'
            (self.output / f'round-{round_id:03d}.json').write_text(json.dumps(
                {'plan': plan, 'reviews': reviews, 'composition': composition, 'subsets': subsets}, indent=2))
            candidates = []
            pending = []
            for subset in subsets:
                if self.evaluated >= self.budget:
                    break
                try:
                    candidate = compose(baseline, [eligible[r] for r in subset])
                    key = self._key(candidate)
                    if key in self._seen:
                        raise ValueError('Previously evaluated pipeline')
                    self.harness.preflight(candidate)
                except Exception as exc:
                    self._record(Outcome(round_id, subset, 'rejected', reason=str(exc)))
                    continue
                self._seen.add(key)
                self.evaluated += 1
                pending.append((candidate, subset))
            for (candidate, subset), result in zip(pending, self._evaluate_batch([x[0] for x in pending])):
                try:
                    if isinstance(result, Exception):
                        raise result
                    if not math.isfinite(result.score):
                        raise ValueError('Non-finite validation score')
                    candidates.append((candidate, result, subset))
                    self._record(Outcome(round_id, subset, 'evaluated', candidate.payload(), result))
                except Exception as exc:
                    self._record(Outcome(round_id, subset, 'failed', candidate.payload(), reason=str(exc)))
            self.stagnant_rounds += 1
            if candidates:
                candidate, result, origin = max(candidates, key=lambda item: item[1].score)
                if result.score > score.score:
                    baseline, score = candidate, result
                    self.improvements += 1
                    self.stagnant_rounds = 0
                    self._record(Outcome(round_id, origin, 'promoted', baseline.payload(), score))
            if self.exploration:
                for row in self.experiment_ledger:
                    if row['round'] != round_id or row.get('historical'):
                        continue
                    outcomes = [o for o in self.history if o.round == round_id and o.origin == (row['role'],)]
                    if outcomes:
                        outcome = outcomes[-1]
                        row.update(status=outcome.status, score=outcome.evaluation.score if outcome.evaluation else None,
                                   reason=outcome.reason)
                (self.output / 'experiment-ledger.json').write_text(json.dumps(self.experiment_ledger, indent=2))

            self._save(baseline, score, round_id)
            with (self.output / 'round-timings.jsonl').open('a') as handle:
                handle.write(json.dumps({'round': round_id, 'elapsed_seconds': time.monotonic() - round_started,
                                         'candidate_evaluations': self.evaluated - evaluated_before}) + '\n')
        if self.stop_reason is None:
            self.stop_reason = ('target_improvements_reached' if self.target_improvements is not None and self.improvements >= self.target_improvements
                                else 'fit_budget_exhausted' if self.evaluated >= self.budget else 'round_limit_reached')
        self._save(baseline, score, min(self.rounds, max(o.round for o in self.history)))
        return baseline, score

    def _save(self, baseline: Pipeline, score: Evaluation, round_id: int):
        from .exploration import failure_memory
        (self.output / 'failure-memory.json').write_text(json.dumps(failure_memory(self.experiment_ledger), indent=2))
        (self.output / 'selected.json').write_text(json.dumps({
            'pipeline': baseline.payload(), 'validation': asdict(score),
            'validation_preprocessing': (self.harness.validation_preprocessing_artifacts(baseline)
                                         if hasattr(self.harness, 'validation_preprocessing_artifacts') else None),
            'architecture_provenance': model_provenance(baseline.model, 'research/components/model.py' in baseline.code),
            'augmentation_provenance': policy_provenance(baseline.augmentation),
            'round': round_id, 'evaluations': self.evaluated,
            'improvements': self.improvements, 'target_improvements': self.target_improvements, 'stop_reason': self.stop_reason,
            'task_fingerprint': self.harness.task.fingerprint,
            'data_identity': (self.harness.task.identity_metadata()
                               if hasattr(self.harness.task, 'identity_metadata') else {'fingerprint_scheme': 'legacy_or_unspecified'}),
        }, indent=2))


class SingleAgentSearch(Search):
    def run(self, initial) -> tuple[Pipeline, Evaluation]:
        baseline, score = self._initialize(initial)
        for round_id in range(1, self.rounds + 1):
            round_started, evaluated_before = time.monotonic(), self.evaluated
            self.stop_reason = self._limit_stop_reason(score)
            if self.stop_reason is not None:
                break
            for _ in range(3):
                self.stop_reason = self._limit_stop_reason(score)
                if self.stop_reason is not None:
                    break
                try:
                    context = self._context(baseline, score, round_id)
                    patches = self.agents.generalist(context)
                    from .exploration import LIMITS
                    sizes = [check_proposal(baseline, p, self.harness.sources, limits=LIMITS) for p in patches]
                    if not sizes:
                        raise ValueError('Empty generalist proposal')
                    limits = LIMITS
                    if any(sum(size[key] for size in sizes) > limit for key, limit in limits.items()):
                        raise ValueError('Generalist mutation exceeds shared bounds')
                    candidate = compose(baseline, patches)
                    key = self._key(candidate)
                    if key in self._seen:
                        raise ValueError('Previously evaluated pipeline')
                    self.harness.preflight(candidate)
                except Exception as exc:
                    self._record(Outcome(round_id, ('generalist',), 'rejected', reason=str(exc)))
                    continue
                self._seen.add(key)
                self.evaluated += 1
                try:
                    result = self.harness.evaluate(candidate)
                    if not math.isfinite(result.score):
                        raise ValueError('Non-finite validation score')
                    self._record(Outcome(round_id, ('generalist',), 'evaluated', candidate.payload(), result))
                    if result.score > score.score:
                        baseline, score = candidate, result
                        self.improvements += 1
                        self._record(Outcome(round_id, ('generalist',), 'promoted', baseline.payload(), score))
                except Exception as exc:
                    self._record(Outcome(round_id, ('generalist',), 'failed', candidate.payload(), reason=str(exc)))
            self._save(baseline, score, round_id)
            with (self.output / 'round-timings.jsonl').open('a') as handle:
                handle.write(json.dumps({'round': round_id, 'elapsed_seconds': time.monotonic() - round_started,
                                         'candidate_evaluations': self.evaluated - evaluated_before}) + '\n')
            if self.stop_reason is not None or self.evaluated >= self.budget:
                break
        if self.stop_reason is None:
            self.stop_reason = ('target_improvements_reached' if self.target_improvements is not None and self.improvements >= self.target_improvements
                                else 'fit_budget_exhausted' if self.evaluated >= self.budget else 'round_limit_reached')
        self._save(baseline, score, min(self.rounds, max(o.round for o in self.history)))
        return baseline, score
