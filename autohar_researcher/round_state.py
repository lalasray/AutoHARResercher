from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class RoundState:
    round_id: int
    proposal_artifacts: dict[str, str] = field(default_factory=dict)
    composed_artifact: str = ""
