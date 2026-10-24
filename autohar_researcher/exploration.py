"""Persistent experimental memory and evidence-directed coverage across families."""
from collections import Counter
import difflib
import hashlib
import json

FAMILIES = {
    'data': ['window_stride', 'fitted_normalization', 'normalization', 'sampling', 'filtering'],
    'model': ['activation', 'architecture', 'representation', 'optimization', 'regularization'],
    'augmentation': ['transform_family', 'magnitude', 'per_class', 'mixing'],
}
LIMITS = {'files': 3, 'functions': 24, 'fields': 64, 'lines': 800}


def family(role, changes, source, rationale=''):
    text = (source + ' ' + rationale).lower()
    keys = set(changes)
    if role == 'data':
        if 'fit_preprocessor' in source: return 'fitted_normalization'
        if keys & {'window_seconds', 'overlap'}: return 'window_stride'
        if 'normalization' in keys: return 'normalization'
        if keys & {'balanced_sampling', 'class_ratios', 'sources'}: return 'sampling'
        return 'filtering'
    if role == 'model':
        if 'architecture' in keys: return 'architecture'
        if keys & {'optimizer','learning_rate','loss','batch_size','max_epochs','patience','scheduler','weight_decay','label_smoothing'} or 'build_optimizer' in text or 'build_loss' in text: return 'optimization'
        if any(x in text for x in ('relu', 'gelu', 'silu', 'activation', 'swish')): return 'activation'
        if 'dropout' in text or 'dropout' in keys or 'head_dropout' in keys: return 'regularization'
        return 'representation'
    if 'per_class' in keys and changes['per_class']: return 'per_class'
    if 'mixing' in keys or any(x in text for x in ('mixup', 'cutmix')): return 'mixing'
    if keys & {'probability','strength','params'} and not source: return 'magnitude'
    return 'transform_family'


def entry(round_id, role, proposal, base, assigned=None):
    changes = proposal.changes
    diff = '\n'.join('\n'.join(difflib.unified_diff(base.code.get(path, '').splitlines(), (source or '').splitlines(), lineterm=''))
                     for path, source in proposal.files.items())
    code = '\n'.join(source or '' for source in proposal.files.values())
    return {'round': round_id, 'role': role, 'family': assigned or family(role, changes, code, proposal.rationale),
            'baseline_context': {**{role: getattr(base,role) for role in FAMILIES}, 'code_sha256': {path: hashlib.sha256(source.encode()).hexdigest() for path,source in base.code.items()}},
            'changes': changes, 'code_sha256': hashlib.sha256(code.encode()).hexdigest(),
            'code_diff': diff, 'rationale': proposal.rationale, 'status': 'proposed'}


def summarize(ledger, stagnation):
    coverage = {role: dict(Counter(e['family'] for e in ledger if e['role'] == role)) for role in FAMILIES}
    options = {}
    for role, families in FAMILIES.items():
        counts = coverage[role]
        minimum = min(counts.get(f, 0) for f in families)
        untried = [f for f in families if counts.get(f, 0) == 0]
        options[role] = untried or ([f for f in families if counts.get(f, 0) == minimum] if stagnation >= 3 else families)
    # Preserve concrete changes and code diffs for older experiments, not just hashes.
    compact = [{**e, 'code_diff': e['code_diff'][:1600], 'code_diff_truncated': len(e['code_diff']) > 1600} for e in ledger]
    return {'stagnant_rounds': stagnation, 'coverage': coverage, 'allowed_focus': options,
            'experiment_ledger': compact, 'failed_ideas': failure_memory(ledger), 'mutation_limits': LIMITS,
            'policy': 'Test an evidence-backed mechanism from an allowed family. Refine successes; after stagnation prioritize least-covered families. One coherent mechanism per candidate; coordinated edits may support it.'}


def validate_plan(plan, exploration):
    hypotheses = plan.get('hypotheses', {})
    for role in FAMILIES:
        item = hypotheses.get(role, {})
        if item.get('family') not in exploration['allowed_focus'][role]:
            raise ValueError(f'{role} plan must select an allowed exploration family')
        for key in ('hypothesis', 'evidence', 'expected_effect', 'falsification'):
            if not isinstance(item.get(key), str) or not item[key].strip():
                raise ValueError(f'{role} plan requires {key}')


def import_history(directory):
    """Recover concrete hypotheses and outcomes from a completed/interrupted search."""
    from .types import Pipeline, Proposal
    history = [json.loads(line) for line in (directory / 'history.jsonl').read_text().splitlines()]
    baseline_rows = [r for r in history if r['status'] == 'baseline']
    if not baseline_rows:
        raise ValueError('History has no validated baseline')
    baseline = Pipeline.from_dict(baseline_rows[0]['pipeline'])
    baseline_score = baseline_rows[0]['evaluation']['score']
    ledger = []
    for path in sorted(directory.glob('round-*.json')):
        round_id = int(path.stem.split('-')[-1])
        report = json.loads(path.read_text())
        for role, review in report.get('reviews', {}).items():
            proposal = Proposal(**review['proposal'])
            row = entry(round_id, role, proposal, baseline)
            row.update(baseline_score=baseline_score, baseline_signature=hashlib.sha256(json.dumps(baseline.payload(),sort_keys=True).encode()).hexdigest())
            outcomes = [r for r in history if r['round'] == round_id and r['origin'] == [role]]
            row.update(historical=True, source=str(directory),
                       status=outcomes[-1]['status'] if outcomes else 'interrupted_or_not_evaluated',
                       score=outcomes[-1]['evaluation']['score'] if outcomes and outcomes[-1].get('evaluation') else None)
            ledger.append(row)
        promoted = [r for r in history if r['round'] == round_id and r['status'] == 'promoted']
        if promoted:
            baseline = Pipeline.from_dict(promoted[-1]['pipeline'])
            baseline_score = promoted[-1]['evaluation']['score']
    return history, ledger


def failure_memory(ledger):
    """Concrete unsuccessful experiments, kept separately from untested rejections."""
    failed = []
    for row in ledger:
        status, score = row.get('status'), row.get('score')
        reference = row.get('baseline_score')
        if status == 'failed' or (status == 'evaluated' and score is not None
                and reference is not None and score <= reference):
            failed.append({key: row.get(key) for key in
                ('round', 'role', 'family', 'changes', 'code_diff', 'rationale',
                 'score', 'baseline_score', 'baseline_signature', 'baseline_context', 'status', 'reason')})
    return failed
