"""Deterministic evidence summaries; full artifacts remain the source of truth."""
from collections import Counter
import hashlib
import json


def compact(value, depth=0):
    if isinstance(value, str):
        return value if len(value) <= 800 else value[:800] + f' ... [{len(value)} characters; details saved]'
    if isinstance(value, dict):
        if depth >= 6:
            return {'available_fields': list(value)[:32], 'summarized': True}
        items = list(value.items())
        result = {key: compact(item, depth + 1) for key, item in items[:32]}
        if len(items) > 32:
            result['omitted_field_count'] = len(items) - 32
        return result
    if isinstance(value, (list, tuple)):
        result = [compact(item, depth + 1) for item in value[:12]]
        if len(value) > 12:
            result.append({'omitted_item_count': len(value) - 12})
        return result
    return value


def diagnostic_summary(data, *, fold_details=True):
    result = {key: data[key] for key in ('mean_validation_macro_f1', 'validation_macro_f1',
              'fold_count', 'scope', 'status', 'reason', 'training_class_counts',
              'validation_class_counts') if key in data}
    classes = data.get('per_class', {})
    if classes:
        result['weakest_classes'] = dict(sorted(classes.items(), key=lambda pair: pair[1]['f1'])[:8])
    if data.get('top_confusions'):
        result['top_confusions'] = data['top_confusions'][:8]
    pipeline = data.get('pipeline', {})
    if pipeline:
        result['architecture'] = pipeline.get('model', {}).get('architecture')
    for section in ('real_training', 'real_validation'):
        values = data.get(section) or {}
        if values:
            result[section] = {key: values[key] for key in ('window_count', 'class_counts') if key in values}
            channels = values.get('channels') or {}
            result[section]['channels'] = {key: channels[key] for key in ('mean', 'std', 'rms') if key in channels}
    for section in ('training_augmentation_probe', 'validation_augmentation_probe'):
        probe = data.get(section) or {}
        if probe:
            result[section] = {'diagnostic_only': True,
                **{key: probe[key] for key in ('error', 'used_for_selection') if key in probe},
                'top_confusions': probe.get('conventional_predictions', {}).get('top_confusions', [])[:3]}
    if fold_details and data.get('folds'):
        folds = []
        for fold in data['folds']:
            if not isinstance(fold, dict):
                continue
            detail = fold.get('diagnostics', {})
            folds.append({'subject': fold['subject'],
                          'validation_macro_f1': detail.get('validation_macro_f1'),
                          'weakest_classes': dict(sorted(detail.get('per_class', {}).items(),
                                                       key=lambda pair: pair[1]['f1'])[:3])})
        result['folds'] = folds[:16]
        result['fold_count'] = len(data['folds'])
        scores = [row['validation_macro_f1'] for row in folds if isinstance(row['validation_macro_f1'], (int, float))]
        if scores:
            result['fold_score_range'] = {'min': min(scores), 'max': max(scores)}
    result['available_sections'] = list(data)
    result['summary_only'] = True
    result['inspection_note'] = 'Request an individual subject and sections for details; large sections are paginated.'
    return compact(result)


def outcome_summary(row):
    result = {key: row[key] for key in ('round', 'origin', 'status', 'reason') if key in row}
    evaluation = row.get('evaluation') or {}
    result['score'] = evaluation.get('score')
    if row.get('pipeline'):
        pipeline = row['pipeline']
        result['pipeline_id'] = hashlib.sha256(json.dumps(pipeline, sort_keys=True).encode()).hexdigest()
        result['architecture'] = pipeline.get('model', {}).get('architecture')
    return compact(result)


def history_summary(rows):
    return {'total_outcomes': len(rows), 'status_counts': dict(Counter(row.get('status') for row in rows)),
            'recent': [outcome_summary(row) for row in rows[-10:]]}


def inspection_summary(data):
    if isinstance(data, dict) and data.get('summary_only'):
        return compact(data)
    if isinstance(data, dict) and any(key in data for key in ('folds', 'per_class', 'mean_validation_macro_f1')):
        return diagnostic_summary(data)
    return compact(data)
