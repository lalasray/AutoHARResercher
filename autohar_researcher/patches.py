"""Deterministic proposal checks, ownership, and compatibility composition."""
from __future__ import annotations

import ast
import copy
import difflib
import importlib.util
import math

from .types import Pipeline, Proposal, ROLES
from research.modules.architectures.presets import model_defaults

OWNED_FILES = {role: f'research/components/{role}.py' for role in ROLES}
ALLOWED_IMPORTS = {'math', 'numpy', 'torch', 'scipy', 'random'}
DATA_FIELDS = {'sources', 'window_seconds', 'overlap', 'normalization', 'balanced_sampling', 'class_ratios'}
MODEL_FIELDS = {'architecture', 'hidden_dim', 'depth', 'dropout', 'optimizer', 'learning_rate',
                'weight_decay', 'loss', 'label_smoothing', 'batch_size', 'max_epochs', 'patience',
                'lstm_layers', 'conv_filters', 'attention_mix', 'bidirectional',
                'conv_channels', 'kernel_size', 'pool_size', 'branch_channels', 'head_dropout',
                'classifier_batch_norm', 'norm_eps', 'batch_norm_momentum',
                'optimizer_eps', 'optimizer_beta1', 'optimizer_beta2', 'n_fft', 'stft_hop', 'optimizer_alpha', 'scheduler', 'lr_patience', 'lr_factor'}
AUGMENTATION_FIELDS = {'kind', 'strength', 'synthetic', 'params', 'probability', 'transforms', 'mixing', 'per_class'}


def validate_pipeline(p: Pipeline, sources: set[str]) -> None:
    for values, fields in ((p.data, DATA_FIELDS), (p.model, MODEL_FIELDS),
                           (p.augmentation, AUGMENTATION_FIELDS)):
        if set(values) - fields:
            raise ValueError(f'Unsupported/protected configuration fields: {set(values) - fields}')
    chosen = p.data['sources'] or sorted(sources)
    if not isinstance(chosen, list) or not chosen or not set(chosen) <= sources:
        raise ValueError('Source subset must contain only eligible source datasets')
    if not 1 <= p.data['window_seconds'] <= 5 or not 0 <= p.data['overlap'] <= .75:
        raise ValueError('Window/overlap outside paper limits')
    if p.data['normalization'] not in {'standard', 'minmax', 'none'}:
        raise ValueError('Normalization must be standard, minmax or none')
    ratios = p.data.get('class_ratios', {})
    if not isinstance(ratios, dict) or any(not math.isfinite(v) or not 0 < v <= 1 for v in ratios.values()):
        raise ValueError('Class sampling ratios must be in (0, 1]')
    if type(p.data['balanced_sampling']) is not bool:
        raise ValueError('balanced_sampling must be boolean')
    if p.model['optimizer'] not in {'adam', 'adamw', 'nadam', 'sgd', 'rmsprop'}:
        raise ValueError('Unsupported optimizer')
    if p.model.get('scheduler', 'none') not in {'none', 'plateau_validation_loss'}:
        raise ValueError('Unsupported learning-rate scheduler')
    if p.model.get('scheduler') == 'plateau_validation_loss':
        if type(p.model.get('lr_patience')) is not int or p.model['lr_patience'] < 1:
            raise ValueError('lr_patience must be a positive integer')
        if not 0 < p.model.get('lr_factor', 0) < 1:
            raise ValueError('lr_factor must be in (0,1)')
    if p.model['loss'] not in {'cross_entropy', 'weighted_cross_entropy'}:
        raise ValueError('Unsupported loss')
    for key in ('batch_size', 'max_epochs', 'patience', 'hidden_dim'):
        if type(p.model[key]) is not int or p.model[key] < 1:
            raise ValueError(f'{key} must be a positive integer')
    for key in ('learning_rate', 'weight_decay', 'dropout', 'label_smoothing'):
        if not math.isfinite(p.model[key]) or p.model[key] < 0:
            raise ValueError(f'Invalid {key}')
    if p.model['learning_rate'] == 0 or not 0 <= p.model['dropout'] < 1 or not 0 <= p.model['label_smoothing'] <= 1:
        raise ValueError('Invalid model regularization/rate')
    for key in ('kernel_size', 'pool_size', 'branch_channels', 'n_fft', 'stft_hop'):
        if key in p.model and (type(p.model[key]) is not int or p.model[key] < 1):
            raise ValueError(f'{key} must be a positive integer')
    for key in ('optimizer_eps', 'norm_eps'):
        if key in p.model and (not math.isfinite(p.model[key]) or p.model[key] <= 0):
            raise ValueError(f'{key} must be positive and finite')
    for key in ('head_dropout', 'batch_norm_momentum', 'optimizer_beta1', 'optimizer_beta2', 'optimizer_alpha'):
        if key in p.model and (not math.isfinite(p.model[key]) or not 0 <= p.model[key] < 1):
            raise ValueError(f'{key} must be in [0, 1)')
    if 'conv_channels' in p.model and (not isinstance(p.model['conv_channels'], list) or
            not p.model['conv_channels'] or any(type(n) is not int or n < 1 for n in p.model['conv_channels'])):
        raise ValueError('conv_channels must be a nonempty list of positive integers')
    if 'classifier_batch_norm' in p.model and type(p.model['classifier_batch_norm']) is not bool:
        raise ValueError('classifier_batch_norm must be boolean')
    from research.augmentations import validate_policy
    validate_policy(p.augmentation)
    if not math.isfinite(p.augmentation['strength']) or p.augmentation['strength'] < 0:
        raise ValueError('Invalid augmentation strength')
    if not isinstance(p.augmentation['synthetic'], list):
        raise ValueError('Synthetic requests must be a list')
    for request in p.augmentation['synthetic']:
        if request['generator'] not in {'text', 'video'}:
            raise ValueError('Synthetic generators must use text or video MCP tools')
        if type(request['count']) is not int or request['count'] < 1:
            raise ValueError('Synthetic count must be positive')
        if not math.isfinite(request.get('ratio', 1)) or not 0 <= request.get('ratio', 1) <= 1:
            raise ValueError('Synthetic mixing ratio must be in [0, 1]')
    if set(p.code) - set(OWNED_FILES.values()):
        raise ValueError('Protected executable file')
    for content in p.code.values():
        validate_code(content)


def validate_code(content: str) -> None:
    tree = ast.parse(content)
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names = [a.name for a in node.names] if isinstance(node, ast.Import) else [node.module or '']
            for name in names:
                root = name.split('.')[0]
                if root not in ALLOWED_IMPORTS or importlib.util.find_spec(root) is None:
                    raise ValueError(f'Unapproved or missing import: {name}')
        if isinstance(node, ast.Name) and node.id in {'open', 'eval', 'exec', 'compile', '__import__', 'globals', 'locals', 'getattr', 'setattr'}:
            raise ValueError(f'Forbidden dynamic access: {node.id}')
        if isinstance(node, ast.Attribute) and ((node.attr.startswith('__') and node.attr != '__init__') or node.attr in {
            'load', 'save', 'loadtxt', 'savetxt', 'fromfile', 'tofile', 'memmap',
            'system', 'popen', 'read', 'write', 'read_text', 'write_text'}):
            raise ValueError(f'Forbidden I/O or introspection: {node.attr}')


def _functions(content: str) -> dict[str, str]:
    return {node.name: ast.dump(node, include_attributes=False)
            for node in ast.walk(ast.parse(content))
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}


def check_proposal(base: Pipeline, proposal: Proposal, sources: set[str]) -> dict[str, int]:
    if proposal.role not in ROLES:
        raise ValueError('Unknown specialist')
    if set(proposal.files) - {OWNED_FILES[proposal.role]}:
        raise ValueError('Specialist attempted to edit a protected or unowned file')
    changes = sum(getattr(base, proposal.role).get(k) != v for k, v in proposal.changes.items())
    lines = functions = 0
    for path, new in proposal.files.items():
        old = base.code.get(path, '')
        validate_code(new)
        lines += sum(line.startswith(('+ ', '- ')) for line in difflib.ndiff(old.splitlines(), new.splitlines()))
        old_f, new_f = _functions(old), _functions(new)
        functions += sum(old_f.get(k) != new_f.get(k) for k in old_f.keys() | new_f.keys())
    if len(proposal.files) > 3 or functions > 3 or changes > 8 or lines > 120:
        raise ValueError('Mutation exceeds 3 files / 3 functions / 8 fields / 120 changed lines')
    if not changes and not lines:
        raise ValueError('Duplicate baseline proposal')
    if set(proposal.dependencies) - set(ROLES):
        raise ValueError('Dependencies must name specialist roles')
    candidate = compose(base, [proposal], check_dependencies=False)
    validate_pipeline(candidate, sources)
    return {'files': len(proposal.files), 'functions': functions, 'fields': changes, 'lines': lines}


def compose(base: Pipeline, proposals: list[Proposal], check_dependencies: bool = True) -> Pipeline:
    result = copy.deepcopy(base)
    roles = {p.role for p in proposals}
    if len(roles) != len(proposals):
        raise ValueError('Multiple patches for the same component')
    touched = set()
    for p in proposals:
        if check_dependencies and not set(p.dependencies) <= roles:
            raise ValueError('Unsatisfied proposal dependencies')
        if touched.intersection(p.files):
            raise ValueError('Overlapping executable edits')
        touched.update(p.files)
        if p.role == 'model' and p.changes.get('architecture', result.model['architecture']) != result.model['architecture']:
            result.model = model_defaults(p.changes['architecture'])
        getattr(result, p.role).update(copy.deepcopy(p.changes))
        result.code.update(p.files)
    return result
