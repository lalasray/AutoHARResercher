"""Explicit recording manifests; no target arrays are opened during search.

Adapters (including WHAR exports) must preserve subject/session boundaries and
provide raw, unnormalized recordings in documented channel order and units.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

import numpy as np

from .types import LODO_DATASETS, LODO_LABELS, LOSO_DATASETS


@dataclass(frozen=True)
class Recording:
    dataset: str
    subject: str
    session: str
    path: Path
    label: str
    hz: float
    channels: tuple[str, ...]
    scales: tuple[float, ...]

    def load(self) -> np.ndarray:
        from scipy.signal import resample_poly
        x = np.load(self.path, allow_pickle=False)
        if x.ndim != 2 or x.shape[1] != len(self.channels) or not np.isfinite(x).all():
            raise ValueError(f'Invalid time × channel recording: {self.path}')
        x = x.astype(np.float64) * np.asarray(self.scales)
        ratio = Fraction(50 / self.hz).limit_denominator(10000)
        # scipy uses a zero-phase low-pass FIR with a Kaiser window.
        return resample_poly(x, ratio.numerator, ratio.denominator, axis=0).astype(np.float32)


@dataclass(frozen=True)
class Task:
    protocol: str
    target: str
    labels: tuple[str, ...]
    channels: tuple[str, ...]
    train: tuple[Recording, ...]
    validation: tuple[Recording, ...]
    test: tuple[Recording, ...]
    split_seed: int
    fingerprint: str

    def public_metadata(self) -> dict:
        # Never expose target paths, samples, counts, or fitted statistics to agents.
        return {'protocol': self.protocol, 'target': self.target, 'labels': self.labels,
                'channels': self.channels, 'sampling_hz': 50, 'fingerprint': self.fingerprint,
                'source_datasets': sorted({r.dataset for r in self.train})}

    def split_manifest(self) -> dict:
        def identities(rows):
            return sorted({(r.dataset, r.subject, r.session) for r in rows})
        return {**self.public_metadata(), 'split_seed': self.split_seed,
                'train': identities(self.train), 'validation': identities(self.validation),
                'test': identities(self.test)}


def load_manifest(path: Path) -> tuple[dict, tuple[Recording, ...]]:
    raw = json.loads(path.read_text())
    records = []
    seen = set()
    sessions = {}
    for row in raw['recordings']:
        dataset = row['dataset']
        spec = raw['datasets'][dataset]
        label = spec['label_map'].get(str(row['label']))
        if label is None:
            raise ValueError(f'Unmapped label {row["label"]!r} in {dataset}')
        channels = tuple(spec['channels'])
        scales = tuple(float(v) for v in spec['unit_scales'])
        hz = float(row.get('sampling_hz', spec['sampling_hz']))
        if len(scales) != len(channels) or hz <= 0 or not np.isfinite([hz, *scales]).all():
            raise ValueError('Invalid sampling rate or unit conversion')
        signal_path = (path.parent / row['path']).resolve()
        if signal_path in seen:
            raise ValueError('A recording path may occur only once')
        seen.add(signal_path)
        key = (dataset, str(row['session']))
        subject = str(row['subject'])
        if key in sessions and sessions[key] != subject:
            raise ValueError('A session cannot span subjects')
        sessions[key] = subject
        records.append(Recording(dataset, subject, str(row['session']), signal_path,
                                 label, hz, channels, scales))
    if not records:
        raise ValueError('Manifest contains no recordings')
    return raw, tuple(records)


def make_task(path: Path, protocol: str, target: str, subject: str | None = None,
              split_seed: int = 13) -> Task:
    raw, recordings = load_manifest(path)
    if protocol == 'loso':
        if target not in LOSO_DATASETS or subject is None:
            raise ValueError('LOSO requires a benchmark dataset and held-out subject')
        rows = [r for r in recordings if r.dataset == target]
        test = [r for r in rows if r.subject == str(subject)]
        source = [r for r in rows if r.subject != str(subject)]
        labels = tuple(raw['datasets'][target]['labels'])
    elif protocol == 'lodo':
        if target not in LODO_DATASETS:
            raise ValueError('LODO targets are UCI-HAR, MotionSense, and RealLife-HAR')
        labels = LODO_LABELS
        # Eligibility is fixed before search. Other datasets in the paper fail
        # channel, placement, correspondence, or acquisition-overlap criteria.
        rows = [r for r in recordings if r.dataset in LODO_DATASETS and r.label in labels]
        test = [r for r in rows if r.dataset == target]
        source = [r for r in rows if r.dataset != target]
        for name in {r.dataset for r in rows}:
            if not raw['datasets'][name].get('placement'):
                raise ValueError('LODO requires documented sensor placement')
        if any(r.channels != ('ax', 'ay', 'az', 'gx', 'gy', 'gz') for r in rows):
            raise ValueError('LODO channel order must be ax, ay, az, gx, gy, gz')
    else:
        raise ValueError('Protocol must be loso or lodo')
    if not source or not test or not labels or len(set(labels)) != len(labels):
        raise ValueError('Empty source/target partition or invalid fixed label space')
    channels = source[0].channels
    if any(r.channels != channels for r in source + test):
        raise ValueError('All recordings must use the fixed channel layout')
    train, validation = [], []
    if protocol == 'loso':
        # One global cycle: every person is test once and validation once.
        # Construct it before removing the test person so folds share the cycle.
        cycle = sorted({r.subject for r in rows})
        if len(cycle) < 3:
            raise ValueError('LOSO requires at least three people for train/validation/test')
        np.random.default_rng(split_seed).shuffle(cycle)
        validation_person = cycle[(cycle.index(str(subject)) + 1) % len(cycle)]
    for dataset in sorted({r.dataset for r in source}):
        subjects = sorted({r.subject for r in source if r.dataset == dataset})
        if len(subjects) < 2:
            raise ValueError(f'{dataset} needs at least two source subjects')
        if protocol == 'loso':
            val_subjects = {validation_person}
        else:
            # LODO retains the paper's per-source 80/20 subject split.
            rng = np.random.default_rng(split_seed)
            rng.shuffle(subjects)
            n_val = max(1, int(np.ceil(0.2 * len(subjects))))
            val_subjects = set(subjects[:n_val])
        for r in source:
            if r.dataset == dataset and r.label in labels:
                (validation if r.subject in val_subjects else train).append(r)
    if not train or not validation:
        raise ValueError('Fixed source split is empty')
    fingerprint = hashlib.sha256(path.read_bytes() +
        f'person-cycle-v2:{protocol}:{target}:{subject}:{split_seed}'.encode()).hexdigest()
    return Task(protocol, target, labels, channels, tuple(train), tuple(validation),
                tuple(test), split_seed, fingerprint)


def windows(records: tuple[Recording, ...] | list[Recording], labels: tuple[str, ...],
            seconds: float, overlap: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not 1 <= seconds <= 5 or not 0 <= overlap <= 0.75:
        raise ValueError('Windows must be 1–5 seconds with 0–75% overlap')
    length = round(seconds * 50)
    stride = max(1, round(length * (1 - overlap)))
    xs, ys, domains = [], [], []
    for r in records:
        if r.label not in labels:
            continue
        x = r.load()
        for start in range(0, len(x) - length + 1, stride):
            xs.append(x[start:start + length])
            ys.append(labels.index(r.label))
            domains.append(r.dataset)
    if not xs:
        raise ValueError('No complete windows in partition')
    return np.stack(xs), np.asarray(ys), np.asarray(domains)
