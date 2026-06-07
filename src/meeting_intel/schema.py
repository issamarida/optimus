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

    @property
    def n_tokens(self) -> int:
        return len(self.text.split())


@dataclass
class Meeting:
    meeting_id: str
    utterances: list[Utterance] = field(default_factory=list)
    category: Optional[str] = None
    client_sentiment: Optional[str] = None
    reference_summary: Optional[str] = None

    @property
    def full_text(self) -> str:
        return "\n".join(u.text for u in self.utterances)

    @property
    def client_text(self) -> str:
        return "\n".join(u.text for u in self.utterances if u.role == ROLE_CLIENT)

    def transcript_lines(self) -> list[str]:
        return [f"[{u.turn_index}] {u.role}/{u.speaker}: {u.text}" for u in self.utterances]
