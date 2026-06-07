from __future__ import annotations

from dataclasses import dataclass, field

from . import config
from .schema import Meeting, ROLE_CLIENT, ROLE_INTERNAL
from .sentiment import SentimentScorer

ACK_TOKENS = ("understand", "sorry", "apolog", "address", "fix", "right away",
              "good question", "let me", "we'll sort", "i hear you")


@dataclass
class CoachingNote:
    kind: str
    observation: str
    suggestion: str
    evidence_turn: int
    evidence_text: str = ""

    def to_dict(self) -> dict:
        return {"kind": self.kind, "observation": self.observation, "suggestion": self.suggestion,
                "evidence_turn": self.evidence_turn, "evidence_text": self.evidence_text}


@dataclass
class CoachingReport:
    notes: list[CoachingNote] = field(default_factory=list)
    method: str = "rule_based"

    def to_dict(self) -> dict:
        return {"method": self.method, "notes": [n.to_dict() for n in self.notes]}


def _following_internal(utterances, i, window):
    return [u for u in utterances[i + 1: i + 1 + window] if u.role == ROLE_INTERNAL]


def _detect(meeting: Meeting, scorer: SentimentScorer, cfg: config.CoachingConfig) -> list[CoachingNote]:
    utterances = meeting.utterances
    notes: list[CoachingNote] = []
    for i, u in enumerate(utterances):
        if u.role != ROLE_CLIENT:
            continue
        following = _following_internal(utterances, i, cfg.response_window)
        if u.is_question and not following:
            notes.append(CoachingNote(
                "unanswered_question",
                "The client asked a direct question that received no reply in the next turns.",
                "Answer client questions before moving on; if unsure, commit to a follow-up time.",
                u.turn_index, u.text))
        elif scorer.predict_one(u.text) == config.NEGATIVE and following and not any(
                tok in t.text.lower() for t in following for tok in ACK_TOKENS):
            notes.append(CoachingNote(
                "unacknowledged_concern",
                "The client expressed a negative sentiment that was not acknowledged in the reply.",
                "Acknowledge the concern explicitly before responding with facts or next steps.",
                u.turn_index, u.text))
    for u in utterances:
        if u.role == ROLE_INTERNAL and u.n_tokens > cfg.monologue_tokens:
            notes.append(CoachingNote(
                "rep_monologue",
                "A long uninterrupted rep turn may have reduced client participation.",
                "Break long explanations into smaller pieces and check in with the client.",
                u.turn_index, u.text[:160]))
    return notes


def assert_grounded(report: CoachingReport) -> None:
    for note in report.notes:
        if note.evidence_turn is None or note.evidence_turn < 0:
            raise ValueError(f"Ungrounded coaching note: {note.kind}")


def coach_meeting(meeting: Meeting, scorer: SentimentScorer | None = None,
                  cfg: config.CoachingConfig = config.CoachingConfig()) -> CoachingReport:
    scorer = scorer or SentimentScorer()
    report = CoachingReport(notes=_detect(meeting, scorer, cfg))
    assert_grounded(report)
    return report
