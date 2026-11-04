"""Reusable local signal helpers; no automatic changes to evaluation protocols.

Prefix calibration needs explicitly supplied initial samples for that person.
These helpers do not fit target-wide statistics or use activity labels. They are
not automatically enabled by the harness's source-only preprocessing interface.
"""
from collections import deque
import math
import numpy as np


CATALOG = {
    'gravity_reference': dict(scope='first supplied sensor samples only',
        findings='Initial 200-sample orientation reference used by the WISDM selected feature expert; no future samples after the prefix.'),
    'rotate_sensor_frame': dict(scope='fixed caller-supplied reference; current window',
        findings='Rotate coherent XYZ triplets into a common gravity frame. Does not append channels or change units/rate. Requires acceleration as the first triplet.'),
    'causal_probability_mean': dict(scope='current and past model predictions only',
        findings='WISDM causal result used history 31 and a reset after 3 consistent predictions with confidence >= .85. Recording boundaries must be independent of activity labels.'),
}


def gravity_reference(initial_samples, sampling_hz, seconds=10.):
    if not math.isfinite(sampling_hz) or sampling_hz <= 0 or not math.isfinite(seconds) or seconds <= 0:
        raise ValueError('Calibration duration and sampling rate must be positive and finite')
    count = round(seconds * sampling_hz)
    x = np.asarray(initial_samples)
    if count < 1 or x.ndim != 2 or x.shape[1] < 3 or len(x) < count:
        raise ValueError('Insufficient initial acceleration samples for calibration')
    prefix = x[:count, :3].astype(np.float32)
    if not np.isfinite(prefix).all():
        raise ValueError('Calibration prefix must be finite')
    mean = prefix.mean(0)
    norm = np.linalg.norm(mean)
    if norm < 1e-6:
        raise ValueError('Cannot infer a gravity direction from a zero-mean prefix')
    return mean / norm


def rotate_sensor_frame(windows, reference):
    x = np.asarray(windows, dtype=np.float32)
    if x.ndim != 3 or x.shape[2] < 3 or x.shape[2] % 3 or not np.isfinite(x).all():
        raise ValueError('Expected finite windows with complete XYZ triplets')
    gravity = np.broadcast_to(np.asarray(reference, dtype=np.float32), (len(x), 3)).copy()
    length = np.linalg.norm(gravity, axis=1, keepdims=True)
    if not np.isfinite(gravity).all() or np.any(length < 1e-6):
        raise ValueError('Gravity reference must be finite and nonzero')
    gravity /= length
    basis = np.tile(np.array([1., 0., 0.], dtype=np.float32), (len(x), 1))
    basis[np.abs(gravity[:, 0]) > .95] = [0, 0, 1]
    first = basis - (basis * gravity).sum(1, keepdims=True) * gravity
    first /= np.linalg.norm(first, axis=1, keepdims=True)
    rotation = np.stack([first, gravity, np.cross(first, gravity)], axis=2)
    return np.einsum('ntgj,njk->ntgk', x.reshape(len(x), x.shape[1], -1, 3), rotation).reshape(x.shape)


class CausalProbabilityMean:
    """Streaming probability smoothing; group keys must not encode activity labels."""
    def __init__(self, history=31, reset_consecutive=3, confidence=.85):
        if type(history) is not int or type(reset_consecutive) is not int or not 1 <= reset_consecutive <= history:
            raise ValueError('Require 1 <= reset_consecutive <= history')
        if not math.isfinite(confidence) or not 0 <= confidence <= 1:
            raise ValueError('Confidence must be in [0,1]')
        self.history, self.consecutive, self.confidence = history, reset_consecutive, confidence
        self.states = {}
        self.classes = None

    def update(self, probabilities, group):
        p = np.asarray(probabilities, dtype=np.float64)
        if p.ndim != 1 or not len(p) or not np.isfinite(p).all() or np.any(p < 0) or not np.isclose(p.sum(), 1.):
            raise ValueError('Expected a finite normalized class-probability vector')
        if self.classes is not None and len(p) != self.classes:
            raise ValueError('Class count cannot change')
        self.classes = len(p)
        history, recent, previous = self.states.get(group, (deque(maxlen=self.history), deque(maxlen=self.consecutive), None))
        history.append(p.copy()); recent.append(p.copy())
        if (len(recent) == self.consecutive and previous is not None
                and all(int(q.argmax()) == int(p.argmax()) and q.max() >= self.confidence for q in recent)
                and int(p.argmax()) != previous):
            history = deque(recent, maxlen=self.history)
        result = np.mean(history, axis=0)
        self.states[group] = history, recent, int(result.argmax())
        return result


def causal_probability_mean(probabilities, groups, history=31, reset_consecutive=3, confidence=.85):
    """Batch convenience wrapper; rows must already be in acquisition order."""
    p = np.asarray(probabilities)
    if p.ndim != 2 or len(p) != len(groups) or not len(p):
        raise ValueError('Expected nonempty aligned probability rows and stream IDs')
    smoother = CausalProbabilityMean(history, reset_consecutive, confidence)
    return np.stack([smoother.update(row, group) for row, group in zip(p, groups)])
