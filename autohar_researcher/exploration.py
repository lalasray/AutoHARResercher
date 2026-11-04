from .context_summary import compact
from collections import Counter
import difflib
import hashlib

FAMILIES = {
    'data': ['window_stride', 'fitted_normalization', 'normalization', 'sampling', 'filtering', 'custom'],
    'model': ['activation', 'architecture', 'representation', 'optimization', 'regularization', 'ensemble', 'cascade', 'custom'],
    'augmentation': ['transform_family', 'magnitude', 'per_class', 'mixing', 'custom'],
}
LIMITS = {'files': 3, 'functions': 128, 'fields': 64, 'lines': 4096}


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

    recent = [{key: compact(value) for key,value in e.items()
               if key != 'baseline_context'} for e in ledger[-12:]]
    failures = [{key: compact(value) for key,value in e.items()
                 if key != 'baseline_context'} for e in failure_memory(ledger)[-12:]]
    return {'stagnant_rounds': stagnation, 'coverage': coverage, 'allowed_focus': FAMILIES, 'preferred_focus': options,
            'experiment_ledger': recent, 'ledger_total': len(ledger), 'failed_ideas': failures, 'mutation_limits': LIMITS,
            'policy': 'Use any coherent mechanism, including custom hooks. Family coverage is advisory; after stagnation prefer less-covered families when justified. One coherent mechanism per candidate; coordinated edits may support it.'}


def validate_plan(plan, exploration):
    hypotheses = plan.get('hypotheses', {})
    for role in FAMILIES:
        item = hypotheses.get(role, {})
        if item.get('family') not in exploration['allowed_focus'][role]:
            raise ValueError(f'{role} plan must select an allowed exploration family')
        for key in ('hypothesis', 'evidence', 'expected_effect', 'falsification'):
            if not isinstance(item.get(key), str) or not item[key].strip():
                raise ValueError(f'{role} plan requires {key}')


def failure_memory(ledger):
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
