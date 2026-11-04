"""Explicit benchmark ontologies and raw CSV adapters that preserve label detail."""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np

WISDM_LABELS = ('walking', 'jogging', 'walking_upstairs', 'walking_downstairs', 'sitting', 'standing')
UNIMIB_ADL9_LABELS = ('sit_to_stand', 'lie_to_stand', 'walking', 'running',
                      'walking_upstairs', 'jumping', 'walking_downstairs', 'stand_to_lie', 'stand_to_sit')
HANGTIME_LABELS = ('dribbling', 'shot', 'pass', 'rebound', 'layup', 'walking', 'running', 'standing', 'sitting')
CAPTURE24_LABELS = ('sleep', 'sitting', 'standing', 'walking', 'bicycling', 'vehicle_travel',
                    'household_chores', 'sports', 'computer_use', 'watching_tv',
                    'stairs_ascent_descent', 'light_activity')
UTD_MHAD_LABELS = (
    'clap', 'throw', 'basketball_shoot', 'drawing', 'boxing', 'baseball_swing',
    'tennis_swing', 'two_hand_push', 'jogging', 'walking', 'squat',
)
LOSO_LABELS = {'unimib_shar': UNIMIB_ADL9_LABELS,
               'utd_mhad': UTD_MHAD_LABELS, 'wisdm': WISDM_LABELS, 'hang_time': HANGTIME_LABELS,
               'capture_24': CAPTURE24_LABELS, 'real_life_har': ('sitting', 'standing', 'walking')}


def subject_order(value):
    """Natural ordering: 2 precedes 10, and P002 precedes P010."""
    return tuple((0, int(part)) if part.isdigit() else (1, part.lower())
                 for part in re.split(r'(\d+)', str(value)) if part)


def select_subjects(subjects, spec):
    available = sorted(set(map(str, subjects)), key=subject_order)
    if 'subjects' in spec and 'first_n_subjects' in spec:
        raise ValueError('Choose subjects or first_n_subjects, not both')
    if 'first_n_subjects' in spec:
        count = spec['first_n_subjects']
        if type(count) is not int or count < 1 or count > len(available):
            raise ValueError('first_n_subjects must be positive and no larger than available subjects')
        return available[:count]
    selected = list(map(str, spec.get('subjects', available)))
    if not selected or len(set(selected)) != len(selected) or set(selected) - set(available):
        raise ValueError('subjects must be unique, nonempty, available subject IDs')
    return sorted(selected, key=subject_order)


def loso_groups(subjects, groups=None):
    """Resolve the LOSO population and test groups; unlisted subjects are excluded."""
    available = set(map(str, subjects))
    if groups is None:
        return [[person] for person in sorted(available, key=subject_order)]
    if not isinstance(groups, list) or not groups:
        raise ValueError('loso_groups must be a nonempty list of subject ID lists')
    result, seen = [], set()
    for group in groups:
        if not isinstance(group, list) or not group or any(type(p) not in (str, int) for p in group):
            raise ValueError('Each LOSO group must be a nonempty subject ID list')
        people = list(map(str, group))
        if len(set(people)) != len(people) or seen.intersection(people):
            raise ValueError('LOSO groups must not contain duplicate or overlapping subjects')
        seen.update(people)
        result.append(people)
    if seen - available:
        raise ValueError('LOSO groups must contain only available subject IDs')
    return result


def capture24_group(annotation):
    """Group activity descriptions into twelve classes.

    Specific context takes precedence over posture. Unlabelled data are excluded;
    remaining descriptive activities enter light_activity. Numeric codes require
    an annotation dictionary and are never interpreted as activity descriptions.
    """
    text = re.sub(r'[_/\-]+', ' ', str(annotation).strip().lower())
    if text in {'', 'nan', 'none', 'unknown', 'unclassified', 'not labeled', 'unlabelled'}:
        return None
    if not re.search(r'[a-z]', text):
        raise ValueError(f'CAPTURE-24 annotation {annotation!r} needs a textual dictionary mapping')
    rules = (
        ('sleep', r'\bsleep\w*\b'),
        ('stairs_ascent_descent', r'\bstairs?\b|\bstairway\w*\b|\bupstairs\b|\bdownstairs\b'),
        ('bicycling', r'\bbicycl\w*\b|\bcycl\w*\b|\bbik\w*\b'),
        ('vehicle_travel', r'\bdriv\w*\b|\bvehicle\w*\b|\bpassenger\w*\b|\bcar\b|\bbus\b|\btrain\b|\bmotor\w*\b'),
        ('computer_use', r'\bcomputer\w*\b|\btyping\b|\bkeyboard\w*\b|\blaptop\w*\b'),
        ('watching_tv', r'\btelevision\b|\btv\b|\bwatching (a )?(movie|film|video)\w*\b'),
        ('household_chores', r'\bclean\w*\b|\bcook\w*\b|\bwash\w*\b|\blaundry\b|\biron\w*\b|\bvacuum\w*\b|\bsweep\w*\b|\bmop\w*\b|\bdishes\b|\bhousework\b|\bhousehold\b|\bgarden\w*\b|\bfood preparation\b'),
        ('sports', r'\bsport\w*\b|\brun\w*\b|\bjog\w*\b|\bswim\w*\b|\bexercise\w*\b|\bdanc\w*\b|\bfootball\b|\btennis\b|\bbasketball\b|\baerobic\w*\b|\bgym\b|\byoga\b'),
        ('walking', r'\bwalk\w*\b'),
        ('sitting', r'\bsit\w*\b|\bseated\b|\bsedentary\b|\blying\b|\breclining\b'),
        ('standing', r'\bstand\w*\b'),
    )
    return next((label for label, pattern in rules if re.search(pattern, text)), 'light_activity')


def hangtime_labels(frame):
    basketball = frame['basketball'].fillna('not_labeled').str.strip().str.lower()
    locomotion = frame['locomotion'].fillna('not_labeled').str.strip().str.lower()
    labels = basketball.where(basketball != 'not_labeled', locomotion)
    unknown = set(labels) - set(HANGTIME_LABELS) - {'jumping', 'not_labeled'}
    if unknown:
        raise ValueError(f'Unknown HangTime labels: {sorted(unknown)}')
    return labels.where(labels.isin(HANGTIME_LABELS) & (frame['coarse'] != 'not_labeled'))


def csv_recordings(raw_dir: Path, spec: dict):
    """Stream raw CAPTURE-24/HangTime CSVs, keeping contiguous runs across chunks."""
    import pandas as pd
    capture = spec['adapter'] == 'capture24_12'
    files = sorted(raw_dir.rglob('P[0-9][0-9][0-9].csv*')) if capture else sorted(raw_dir.rglob('*.csv'))
    if not files:
        raise ValueError(f'No participant CSV files found in {raw_dir}')
    by_subject = {}
    for path in files:
        subject = path.name.split('.')[0]
        if subject in by_subject:
            raise ValueError(f'Duplicate participant file for {subject}')
        by_subject[subject] = path
    selected = select_subjects(by_subject, spec)
    annotation_map = {}
    dictionary = {}
    if capture and spec.get('annotation_dictionary'):
        dictionary_path = raw_dir / spec['annotation_dictionary']
        df = pd.read_csv(dictionary_path)
        key, description = spec.get('dictionary_key', 'annotation'), spec.get('dictionary_description', 'annotation')
        dictionary = dict(zip(df[key].astype(str), df[description].astype(str)))
    session = 0
    for subject in selected:
        pending, previous_label, previous_time, previous_coarse = [], None, None, None
        for frame in pd.read_csv(by_subject[subject], chunksize=250_000):
            values = frame[spec['source_columns']].to_numpy(dtype=np.float32)
            time = frame[spec.get('timestamp_column', 'time' if capture else 'timestamp')]
            times = pd.to_datetime(time, unit=spec.get('timestamp_unit'), errors='raise')
            if capture:
                raw_labels = frame[spec.get('annotation_column', 'annotation')].fillna('').astype(str)
                for raw in raw_labels.unique():
                    if raw not in annotation_map:
                        description = dictionary.get(raw, raw)
                        overrides = spec.get('annotation_overrides', {})
                        annotation_map[raw] = overrides[raw] if raw in overrides else capture24_group(description)
                labels = raw_labels.map(annotation_map)
                coarse = [''] * len(frame)
            else:
                labels = hangtime_labels(frame)
                coarse = frame['coarse'].tolist()
            # Boundaries are computed before exclusions so separated runs never join.
            labels = np.array([None if pd.isna(label) else label for label in labels], dtype=object)
            labels[~np.isfinite(values).all(axis=1)] = None
            if times.isna().any():
                raise ValueError('Missing timestamp in participant recording')
            if not len(frame):
                continue
            coarse = np.asarray(coarse, dtype=object)
            deltas = times.diff().dt.total_seconds().to_numpy(copy=True)
            deltas[0] = ((times.iloc[0] - previous_time).total_seconds()
                         if previous_time is not None else 1 / spec['sampling_hz'])
            prior_labels = np.r_[np.array([previous_label], dtype=object), labels[:-1]]
            prior_coarse = np.r_[np.array([previous_coarse], dtype=object), coarse[:-1]]
            boundaries = ((labels != prior_labels) | (coarse != prior_coarse)
                          | (deltas <= 0) | (deltas > 1.5 / spec['sampling_hz']))
            starts = np.unique(np.r_[0, np.flatnonzero(boundaries)])
            for start, end in zip(starts, np.r_[starts[1:], len(frame)]):
                if boundaries[start] and pending:
                    yield subject, str(session), previous_label, np.concatenate(pending)
                    session += 1
                    pending = []
                previous_label = labels[start]
                if previous_label is not None:
                    pending.append(values[start:end].copy())
            previous_time, previous_coarse = times.iloc[-1], coarse[-1]
        if pending:
            yield subject, str(session), previous_label, np.concatenate(pending)
            session += 1
