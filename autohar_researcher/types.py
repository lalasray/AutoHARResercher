from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from research.modules.architectures.presets import model_defaults

ROLES = ('data', 'model', 'augmentation')
SEEDS = (13, 37, 73, 101, 131)
LOSO_DATASETS = ('wisdm', 'uci_har', 'utd_mhad', 'hapt', 'usc_had',
                 'unimib_shar', 'motion_sense', 'real_life_har', 'hang_time', 'capture_24')
LODO_DATASETS = ('uci_har', 'motion_sense', 'real_life_har')
LODO_LABELS = ('walking', 'walking_upstairs', 'walking_downstairs', 'sitting', 'standing')


@dataclass
class Pipeline:
    # Reference template for explicit configurations; automatic search uses the dataset registry.
    data: dict[str, Any] = field(default_factory=lambda: {
        'sources': [], 'window_seconds': 2.56, 'overlap': 0.5,
        'normalization': 'standard', 'balanced_sampling': False,
    })
    model: dict[str, Any] = field(default_factory=model_defaults)
    augmentation: dict[str, Any] = field(default_factory=lambda: {
        'kind': 'none', 'strength': 0.02, 'synthetic': [],
    })
    # Candidate-owned executable files; never edits the evaluator's checkout.
    code: dict[str, str] = field(default_factory=dict)

    def payload(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict) -> Pipeline:
        if set(value) - {'data', 'model', 'augmentation', 'code'}:
            raise ValueError('Unknown pipeline sections')
        result = cls()
        if 'architecture' in value.get('model', {}):
            result.model = model_defaults(value['model']['architecture'])
        for name in ('data', 'model', 'augmentation', 'code'):
            if name in value:
                getattr(result, name).update(value[name])
        return result


@dataclass(frozen=True)
class Proposal:
    role: str
    changes: dict[str, Any]
    files: dict[str, str]
    rationale: str
    dependencies: tuple[str, ...] = ()
    validation: tuple[str, ...] = ()


@dataclass(frozen=True)
class Judgment:
    novelty: int
    plausibility: int
    compatibility: int
    rationale: str = ''

    @property
    def eligible(self) -> bool:
        scores = (self.novelty, self.plausibility, self.compatibility)
        return (all(type(s) is int and 1 <= s <= 5 for s in scores)
                and self.novelty >= 2 and self.plausibility >= 3
                and self.compatibility >= 4)


@dataclass(frozen=True)
class Evaluation:
    score: float
    dataset_scores: dict[str, float]
    best_epoch: int


@dataclass
class Outcome:
    round: int
    origin: tuple[str, ...]
    status: str
    pipeline: dict | None = None
    evaluation: Evaluation | None = None
    reason: str = ''
