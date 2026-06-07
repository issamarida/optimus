"""Data contract.

The first practitioner question is the prediction problem, not the cleaning. This
module pins down the unit of observation and, critically, *where labels live* so
it is structurally impossible to feed a label in as a feature.

Unit of observation: an ``Utterance`` is one speaker turn; a ``Meeting`` is an
ordered list of utterances plus metadata.

Available at prediction time: transcript text, speaker attribution, speaker role
(client vs internal) when the source provides it, and turn order. Optional timing
is ``None`` when absent — that absence is information, not a zero to impute.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

ROLE_CLIENT = "client"
ROLE_INTERNAL = "internal"
ROLE_UNKNOWN = "unknown"
ROLES = (ROLE_CLIENT, ROLE_INTERNAL, ROLE_UNKNOWN)


@dataclass
class Utterance:
    speaker: str
    text: str
    role: str = ROLE_UNKNOWN
    turn_index: int = 0
    start_sec: Optional[float] = None
    end_sec: Optional[float] = None

    def __post_init__(self) -> None:
        if self.role not in ROLES:
            self.role = ROLE_UNKNOWN

    @property
    def is_question(self) -> bool:
        return "?" in self.text


@dataclass
class Meeting:
    meeting_id: str
    utterances: list[Utterance] = field(default_factory=list)
    # Labels live OUTSIDE the feature-bearing structure on purpose.
    category: Optional[str] = None           # supervised target (discussion category)
    client_sentiment: Optional[str] = None   # validation label for sentiment
    reference_summary: Optional[str] = None   # gold summary for ROUGE, when available

    @property
    def full_text(self) -> str:
        return "\n".join(u.text for u in self.utterances)

    @property
    def client_text(self) -> str:
        return "\n".join(u.text for u in self.utterances if u.role == ROLE_CLIENT)

    def transcript_lines(self) -> list[str]:
        return [f"[{u.turn_index}] {u.role}/{u.speaker}: {u.text}" for u in self.utterances]
