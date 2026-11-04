from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace
from fractions import Fraction
from pathlib import Path

import numpy as np

from .benchmark_data import LOSO_LABELS, loso_groups
from .types import LODO_DATASETS, LODO_LABELS, LOSO_DATASETS
from .data_identity import (FINGERPRINT_SCHEME, digest_json, file_signature, recording_checksum,
                            recording_metadata, partition_identity, task_fingerprint)


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
    content_sha256: str | None = None
    content_signature: tuple | None = None

    def load(self, target_hz: float = 50) -> np.ndarray:
        from scipy.signal import resample_poly
        if self.content_signature is not None and file_signature(self.path) != self.content_signature:
            raise ValueError(f'Recording changed after dataset identity verification; reload the dataset snapshot: {self.path}')
        x = np.load(self.path, allow_pickle=False)
        if self.content_signature is not None and file_signature(self.path) != self.content_signature:
            raise ValueError(f'Recording changed while loading: {self.path}')
        if x.ndim != 2 or x.shape[1] != len(self.channels) or not np.isfinite(x).all():
            raise ValueError(f'Invalid time × channel recording: {self.path}')
        x = x.astype(np.float64) * np.asarray(self.scales)
        ratio = Fraction(target_hz / self.hz).limit_denominator(10000)

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
    sampling_hz: float = 50
    data_fingerprint: str | None = None
    split_fingerprint: str | None = None
    manifest_sha256: str | None = None
    fingerprint_scheme: str = 'legacy_or_unspecified'

    def identity_metadata(self):
        return dict(fingerprint_scheme=self.fingerprint_scheme, data_fingerprint=self.data_fingerprint,
                    split_fingerprint=self.split_fingerprint, manifest_sha256=self.manifest_sha256)

    def public_metadata(self) -> dict:

        return {'protocol': self.protocol, 'target': self.target, 'labels': self.labels,
                'channels': self.channels, 'sampling_hz': self.sampling_hz, 'fingerprint': self.fingerprint,
                **self.identity_metadata(),
                'source_datasets': sorted({r.dataset for r in self.train})}

    def split_manifest(self) -> dict:
        def identities(rows):
            return sorted({(r.dataset, r.subject, r.session) for r in rows})
        return {**self.public_metadata(), 'split_seed': self.split_seed,
                'train': identities(self.train), 'validation': identities(self.validation),
                'test': identities(self.test)}


def load_manifest(path: Path) -> tuple[dict, tuple[Recording, ...]]:
    return _parse_manifest(path, path.read_bytes())


def _parse_manifest(path, manifest_bytes):
    raw = json.loads(manifest_bytes)
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


@dataclass(frozen=True)
class DatasetSnapshot:
    """Run-scoped content verification, shared across task creation only.

    Inputs must remain immutable during the run. A new setup always hashes
    actual bytes again, including when file size and timestamps are unchanged.
    """
    manifest_path: Path
    manifest_signature: tuple
    raw: dict
    recordings: tuple[Recording, ...]
    data_fingerprint: str
    manifest_sha256: str
    checksum_ledger: tuple[dict, ...]

    def provenance(self):
        return dict(fingerprint_scheme=FINGERPRINT_SCHEME, checksum_algorithm='sha256',
                    checksum_scope='complete referenced recording file bytes',
                    verification='streamed once during run setup; inputs must remain immutable',
                    data_fingerprint=self.data_fingerprint, manifest_sha256=self.manifest_sha256,
                    recordings=list(self.checksum_ledger))


def load_dataset_snapshot(path: Path) -> DatasetSnapshot:
    path = path.resolve()
    signature = file_signature(path)
    manifest_bytes = path.read_bytes()
    raw, recordings = _parse_manifest(path, manifest_bytes)
    checked, ledger, content_entries = [], [], []
    for recording, row in zip(recordings, raw['recordings']):
        checksum, recording_signature = recording_checksum(recording.path)
        checked.append(replace(recording, content_sha256=checksum, content_signature=recording_signature))
        entry = dict(**recording_metadata(recording), sha256=checksum)
        content_entries.append(entry)
        ledger.append(dict(**entry, path=row['path'], size_bytes=recording_signature[2]))
    if file_signature(path) != signature:
        raise ValueError('Manifest changed during dataset identity verification')
    if any(file_signature(r.path) != r.content_signature for r in checked):
        raise ValueError('Recording changed during dataset identity verification')
    ordered = sorted(content_entries, key=lambda row: json.dumps(row, sort_keys=True))
    identity = digest_json(dict(scheme='interpreted-recording-collection-v1',
                                processing_hz=raw.get('processing_hz', 50), recordings=ordered))
    return DatasetSnapshot(path, signature, raw, tuple(checked), identity,
                           hashlib.sha256(manifest_bytes).hexdigest(), tuple(ledger))


def _snapshot(path, snapshot):
    if snapshot is None:
        return load_dataset_snapshot(path)
    if path.resolve() != snapshot.manifest_path or file_signature(path) != snapshot.manifest_signature:
        raise ValueError('Dataset snapshot does not match the current manifest; reload it')
    if any(file_signature(r.path) != r.content_signature for r in snapshot.recordings):
        raise ValueError('Recording changed after dataset identity verification; reload the dataset snapshot')
    return snapshot


def make_task(path: Path, protocol: str, target: str, subject: str | None = None,
              split_seed: int = 13, *, grouped: bool = True, snapshot: DatasetSnapshot | None = None) -> Task:
    snapshot = _snapshot(path, snapshot)
    raw, recordings = snapshot.raw, snapshot.recordings
    sampling_hz = raw.get('processing_hz', 50)
    if type(sampling_hz) not in (int, float) or not np.isfinite(sampling_hz) or sampling_hz <= 0:
        raise ValueError('processing_hz must be a finite positive number')
    if protocol == 'loso':
        if target not in LOSO_DATASETS or subject is None:
            raise ValueError('LOSO requires a benchmark dataset and held-out subject')
        labels = LOSO_LABELS.get(target, tuple(raw['datasets'][target]['labels']))
        rows = [r for r in recordings if r.dataset == target and r.label in labels]
        spec = raw['datasets'][target]
        configured_groups = loso_groups({r.subject for r in rows}, spec.get('loso_groups'))
        selected_people = {person for group in configured_groups for person in group}
        rows = [r for r in rows if r.subject in selected_people]
        groups = configured_groups if grouped else loso_groups(selected_people)
        held_out = next((group for group in groups if str(subject) in group), [])
        test = [r for r in rows if r.subject in held_out]
        source = [r for r in rows if r.subject not in held_out]
        subject = ','.join(held_out)
    elif protocol == 'lodo':
        if target not in LODO_DATASETS:
            raise ValueError('LODO targets are UCI-HAR, MotionSense, and RealLife-HAR')
        labels = LODO_LABELS
        if sampling_hz != 50:
            raise ValueError('LODO requires processing_hz=50')
        missing = set(LODO_DATASETS) - set(raw['datasets'])
        if missing:
            raise ValueError(f'LODO requires all three datasets; missing: {sorted(missing)}')
        for name in LODO_DATASETS:
            declared = set(raw['datasets'][name]['labels'])
            observed = {r.label for r in recordings if r.dataset == name}
            if not (set(labels) & declared & observed):
                raise ValueError(f'LODO dataset {name} has no recordings in the fixed activity vocabulary')


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
    if protocol == 'loso' and len({r.subject for r in rows}) < 3:
        raise ValueError('LOSO requires at least three people for train/validation/test')
    for dataset in sorted({r.dataset for r in source}):
        subjects = sorted({r.subject for r in source if r.dataset == dataset})
        if len(subjects) < 2:
            raise ValueError(f'{dataset} needs at least two source subjects')

        rng = np.random.default_rng(split_seed)
        rng.shuffle(subjects)
        n_val = max(1, int(np.ceil(0.2 * len(subjects))))
        val_subjects = set(subjects[:n_val])
        for r in source:
            if r.dataset == dataset and r.label in labels:
                (validation if r.subject in val_subjects else train).append(r)
    if not train or not validation:
        raise ValueError('Fixed source split is empty')
    split = partition_identity(protocol, target, labels, channels, sampling_hz, split_seed, train, validation, test)
    fingerprint = task_fingerprint(snapshot.data_fingerprint, split, snapshot.manifest_sha256)
    return Task(protocol, target, labels, channels, tuple(train), tuple(validation),
                tuple(test), split_seed, fingerprint, sampling_hz, snapshot.data_fingerprint,
                split, snapshot.manifest_sha256, FINGERPRINT_SCHEME)


def make_fixed_validation_task(path: Path, target: str, subject: str | None = None,
                               split_seed: int = 13, *, snapshot: DatasetSnapshot | None = None) -> Task:
    snapshot = _snapshot(path, snapshot)
    raw, recordings = snapshot.raw, snapshot.recordings
    if target not in LOSO_DATASETS:
        raise ValueError('Fixed validation requires a benchmark dataset with at least three people')
    available = {r.subject for r in recordings if r.dataset == target}
    groups = loso_groups(available, raw['datasets'][target].get('loso_groups'))
    people = sorted({person for group in groups for person in group})
    if target not in LOSO_DATASETS or len(people) < 3:
        raise ValueError('Fixed validation requires a benchmark dataset with at least three people')
    if subject is None:
        np.random.default_rng(split_seed).shuffle(people)
        subject = people[0]
    if str(subject) not in people:
        raise ValueError('Unknown validation subject')


    base = make_task(path, 'loso', target, str(subject), split_seed, grouped=False, snapshot=snapshot)
    split = partition_identity('fixed-validation', target, base.labels, base.channels,
                                base.sampling_hz, split_seed, base.train + base.validation, base.test, ())
    fingerprint = task_fingerprint(snapshot.data_fingerprint, split, snapshot.manifest_sha256)
    return Task('fixed-validation', target, base.labels, base.channels,
                base.train + base.validation, base.test, (), split_seed,
                fingerprint, base.sampling_hz, snapshot.data_fingerprint, split,
                snapshot.manifest_sha256, FINGERPRINT_SCHEME)


def windows(records: tuple[Recording, ...] | list[Recording], labels: tuple[str, ...],
            seconds: float, overlap: float, *, sampling_hz: float = 50, return_subjects: bool = False) -> tuple[np.ndarray, ...]:
    if not .5 <= seconds <= 10 or not 0 <= overlap <= 0.95:
        raise ValueError('Windows must be 0.5–10 seconds with 0–95% overlap')
    if type(sampling_hz) not in (int, float) or not np.isfinite(sampling_hz) or sampling_hz <= 0:
        raise ValueError('sampling_hz must be a finite positive number')
    length = round(seconds * sampling_hz)
    stride = max(1, round(length * (1 - overlap)))
    xs, ys, domains, subjects = [], [], [], []
    for r in records:
        if r.label not in labels:
            continue
        x = r.load(sampling_hz)
        for start in range(0, len(x) - length + 1, stride):
            xs.append(x[start:start + length])
            ys.append(labels.index(r.label))
            domains.append(r.dataset)
            if return_subjects:
                subjects.append(r.subject)
    if not xs:
        raise ValueError('No complete windows in partition')
    result = (np.stack(xs), np.asarray(ys), np.asarray(domains))
    return (*result, np.asarray(subjects)) if return_subjects else result
