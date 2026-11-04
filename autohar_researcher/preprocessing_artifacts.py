"""Candidate-specific validation preprocessing evidence and legacy recovery."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile

from .types import Pipeline


def pipeline_id(pipeline):
    return hashlib.sha256(json.dumps(pipeline.payload(), sort_keys=True).encode()).hexdigest()


def snapshot_path(output, pipeline):
    return Path(output) / 'validation-preprocessing' / (pipeline_id(pipeline) + '.pt')


def _atomic_save(payload, destination):
    import torch
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=destination.parent, suffix='.tmp', delete=False) as handle:
        temporary = Path(handle.name)
    try:
        torch.save(payload, temporary)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


def save_validation_snapshot(output, pipeline, payload, *, task_fingerprint, seed):
    """Write a candidate snapshot plus the explicitly non-authoritative latest copy."""
    result = deepcopy(payload)
    result.pop('loaded_artifact', None)
    result.pop('recovery_scope', None)
    if not {'pipeline', 'state', 'normalization_mean', 'normalization_scale'} <= result.keys():
        raise ValueError('Incomplete validation preprocessing snapshot')
    if result.get('fit_scope', 'source_training_only') != 'source_training_only':
        raise ValueError('Cannot relabel final-refit parameters as validation preprocessing')
    if result['pipeline'] != pipeline.payload():
        raise ValueError('Preprocessing snapshot belongs to another pipeline')
    if result.get('task_fingerprint', task_fingerprint) != task_fingerprint or result.get('seed', seed) != seed:
        raise ValueError('Preprocessing snapshot context mismatch')
    result.update(artifact_schema_version=1, pipeline_id=pipeline_id(pipeline),
                  task_fingerprint=task_fingerprint, seed=seed,
                  fit_scope='source_training_only', artifact_role='candidate_validation_preprocessing')
    path = snapshot_path(output, pipeline)
    result['canonical_artifact'] = path.relative_to(output).as_posix()
    _atomic_save(result, path)
    latest = {**result, 'artifact_role': 'latest_validation_evaluation_compatibility_copy'}
    _atomic_save(latest, Path(output) / 'preprocessing-state.pt')
    return result


def load_validation_snapshot(output, pipeline, *, task_fingerprint, seed):
    """Recover only matching validation evidence; never use final-refit parameters.

    Trusted local PyTorch artifacts may contain NumPy arrays. Old snapshots did
    not store task/seed provenance; expose that limitation instead of inventing it.
    """
    import torch
    path = snapshot_path(output, pipeline)
    legacy = not path.exists()
    if legacy:
        path = Path(output) / 'preprocessing-state.pt'
    if not path.exists():
        raise FileNotFoundError('No saved validation preprocessing snapshot for this candidate')
    result = torch.load(path, map_location='cpu', weights_only=False)
    required = {'pipeline', 'state', 'normalization_mean', 'normalization_scale'}
    if not isinstance(result, dict) or not required <= result.keys():
        raise ValueError('Incomplete validation preprocessing snapshot; explicit state is required even when None')
    if pipeline_id(Pipeline.from_dict(result['pipeline'])) != pipeline_id(pipeline):
        raise ValueError('Saved preprocessing snapshot belongs to another pipeline; matching evidence is unavailable')
    if result.get('pipeline_id', pipeline_id(pipeline)) != pipeline_id(pipeline):
        raise ValueError('Saved preprocessing pipeline identifier mismatch')
    if result.get('fit_scope') != 'source_training_only':
        raise ValueError('Expected validation preprocessing fitted on source training only')
    if result.get('task_fingerprint', task_fingerprint) != task_fingerprint or result.get('seed', seed) != seed:
        raise ValueError('Saved preprocessing snapshot context mismatch')
    return {**result, 'loaded_artifact': path.relative_to(output).as_posix(),
            'recovery_scope': ('legacy_snapshot_without_task_or_seed_provenance'
                               if 'task_fingerprint' not in result or 'seed' not in result
                               else 'candidate_and_context_verified')}
