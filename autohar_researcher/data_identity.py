"""Streaming recording checksums. No persistent timestamp-only checksum cache."""
import hashlib
import json


FINGERPRINT_SCHEME = 'recording-content-and-split-v1'


def digest_json(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     ensure_ascii=True, allow_nan=False).encode()).hexdigest()


def file_signature(path):
    value = path.stat()
    return (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns)


def recording_checksum(path):
    """Hash actual file bytes with bounded memory, rejecting concurrent changes."""
    before = file_signature(path)
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    if file_signature(path) != before:
        raise ValueError(f'Recording changed while calculating its checksum: {path}')
    return digest.hexdigest(), before


def recording_metadata(recording):
    return dict(dataset=recording.dataset, subject=recording.subject,
                session=recording.session, label=recording.label,
                sampling_hz=recording.hz, channels=recording.channels,
                unit_scales=recording.scales)


def partition_identity(protocol, target, labels, channels, sampling_hz, split_seed,
                       train, validation, test):
    def identities(rows):
        return sorted((r.dataset, r.subject, r.session, r.label) for r in rows)
    return digest_json(dict(scheme='subject-partitions-v1', protocol=protocol, target=target,
        labels=labels, channels=channels, sampling_hz=sampling_hz, split_seed=split_seed,
        train=identities(train), validation=identities(validation), test=identities(test)))


def task_fingerprint(data_fingerprint, split_fingerprint, manifest_sha256=None):
    return digest_json(dict(scheme=FINGERPRINT_SCHEME, data_fingerprint=data_fingerprint,
                            split_fingerprint=split_fingerprint, manifest_sha256=manifest_sha256))
