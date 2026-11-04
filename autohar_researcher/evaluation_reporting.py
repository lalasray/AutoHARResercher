"""Reporting only: never fits models, reads target signals, or selects pipelines."""
from copy import deepcopy
import json
from pathlib import Path

import numpy as np


def loso_test_statistics(reports):
    """Keep the primary fold mean distinct from descriptive subject statistics."""
    if not reports:
        raise ValueError('LOSO reporting requires at least one completed fold')
    values = [float(row['test_macro_f1']) for row in reports]
    if not np.isfinite(values).all():
        raise ValueError('Fold scores must be finite')
    subjects = {}
    missing = []
    memberships_known = True
    seen = set()
    for row in reports:
        people = row.get('test_subjects')
        if people is None:
            memberships_known = False
            continue
        people = list(map(str, people))
        if not people:
            raise ValueError('A test fold must contain at least one subject')
        if len(set(people)) != len(people) or seen.intersection(people):
            raise ValueError('Test subject identities must not overlap between folds')
        seen.update(people)
        scores = row.get('test_subject_scores', {})
        if set(scores) - set(people):
            raise ValueError('Subject scores must belong to the held-out fold')
        for person in people:
            if person in scores:
                value = float(scores[person])
                if not np.isfinite(value):
                    raise ValueError('Subject scores must be finite')
                subjects[person] = value
            else:
                missing.append(person)
    complete = memberships_known and not missing
    subject_values = list(subjects.values())
    return {
        'reporting_schema_version': 2,
        # Stable primary metric consumed by existing CLI summaries and artifacts.
        'test_macro_f1': float(np.mean(values)),
        'test_std_across_folds': float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
        'tested_fold_count': len(values),
        'test_metric_scope': {
            'test_macro_f1': 'unweighted arithmetic mean of held-out fold Macro-F1 scores',
            'test_std_across_folds': 'sample standard deviation of held-out fold Macro-F1 scores',
            'within_fold': 'pool windows from all held-out subjects before computing fixed-class Macro-F1',
            'fold_weighting': 'equal, regardless of subject count or window count',
            'std_ddof': 1,
            'singleton_std_convention': 0.0,
        },
        'test_subject_statistics': {
            'status': 'complete' if complete else 'unavailable',
            'scope': 'descriptive individual-subject Macro-F1 from each subject\'s held-out fold model; equal subject weights',
            'class_scope': 'same fixed task label space as the fold metric; absent classes retain zero F1',
            'mean': float(np.mean(subject_values)) if complete else None,
            'std': (float(np.std(subject_values, ddof=1)) if len(subject_values) > 1 else 0.0) if complete else None,
            'std_ddof': 1,
            'singleton_std_convention': 0.0,
            'scored_subject_count': len(subject_values),
            'expected_subject_count': len(seen) if memberships_known else None,
            'scores': subjects,
            'missing_subjects': missing,
            'fold_membership_known': memberships_known,
            'reason': None if complete else 'Per-subject predictions/scores or fold memberships were not saved for every test subject; cannot infer subject variation from pooled fold scores.',
        },
    }


def normalize_loso_report(report):
    """Read legacy results without rewriting them or fabricating subject scores.

    Legacy ``test_std_across_subjects`` measured fold variation. Recompute its
    correctly named replacement from the saved final_loso rows, not from any
    target data. Existing primary values and fold artifacts remain unchanged.
    """
    result = deepcopy(report)
    statistics = loso_test_statistics(result['final_loso'])
    if not np.isclose(float(result['test_macro_f1']), statistics['test_macro_f1'], rtol=1e-10, atol=1e-12):
        raise ValueError('Saved primary score does not match the unweighted fold mean')
    legacy = result.pop('test_std_across_subjects', None)
    result.update(statistics)
    # Preserve the stored primary value, including historical serialization.
    result['test_macro_f1'] = report['test_macro_f1']
    if legacy is not None:
        result['legacy_reporting_note'] = 'The removed test_std_across_subjects field was fold-level variability, not individual-subject variability.'
    return result


def load_loso_report(path: Path):
    """Load a legacy or current report into schema 2; source files stay intact."""
    return normalize_loso_report(json.loads(Path(path).read_text(encoding='utf-8')))
