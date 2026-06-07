"""Evidence-grounded coaching feedback.

DESIGN STANCE (read this before trusting any output)
-----------------------------------------------------
There is no ground-truth label for "good coaching", so this is NOT a trained,
measured model — it is advisory decision-support. To keep it honest rather than a
hallucinated opinion generator, every suggestion must be ANCHORED to a concrete,
deterministically detected moment in the transcript, with the turn index. We do
not make counterfactual causal claims ("the client would have been happier") —
we surface observable patterns and constructive alternatives the rep can judge.

The detectors below are event detectors (specific coachable moments tied to a
turn), not the participant "engagement score" that was removed from this project.

Honesty guard: ``coach_meeting`` returns only items that carry an ``evidence_turn``;
``assert_grounded`` enforces that nothing free-floats.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .schema import Meeting, ROLE_CLIENT, ROLE_INTERNAL
from .sentiment import SentimentScorer

_ACK_TOKENS = ("understand", "sorry", "apolog", "address", "fix", "right away",
               "good question", "let me", "we'll sort", "i hear you")
_MONOLOGUE_TOKENS = 45  # an internal turn longer than this is a candidate monologue


@dataclass
class CoachingNote:
    kind: str                 # what pattern fired
    observation: str          # neutral description of what happened
    suggestion: str           # constructive alternative
    evidence_turn: int        # index into the transcript — always present
    evidence_text: str = ""

    def to_dict(self) -> dict:
        return {"kind": self.kind, "observation": self.observation,
                "suggestion": self.suggestion, "evidence_turn": self.evidence_turn,
                "evidence_text": self.evidence_text}


@dataclass
class CoachingReport:
    notes: list[CoachingNote] = field(default_factory=list)
    method: str = "rule_based"

    def to_dict(self) -> dict:
        return {"method": self.method, "notes": [n.to_dict() for n in self.notes]}


def _next_internal_turns(utts, i, window):
    return [u for u in utts[i + 1: i + 1 + window] if u.role == ROLE_INTERNAL]


def _detect(meeting: Meeting, scorer: SentimentScorer, window: int = 2) -> list[CoachingNote]:
    utts = meeting.utterances
    notes: list[CoachingNote] = []
    for i, u in enumerate(utts):
        if u.role != ROLE_CLIENT:
            continue
        following = _next_internal_turns(utts, i, window)

        # Pattern 1: client question with no internal response in the window.
        if u.is_question and not following:
            notes.append(CoachingNote(
                kind="unanswered_question",
                observation="The client asked a direct question that received no reply in the next turns.",
                suggestion="Answer client questions before moving on; if you don't know, say when you'll follow up.",
                evidence_turn=u.turn_index, evidence_text=u.text))

        # Pattern 2: negative client turn that no internal turn acknowledges.
        elif scorer.predict_one(u.text) == "negative":
            acknowledged = any(any(tok in t.text.lower() for tok in _ACK_TOKENS) for t in following)
            if following and not acknowledged:
                notes.append(CoachingNote(
                    kind="unacknowledged_concern",
                    observation="The client expressed a negative sentiment that wasn't acknowledged in the reply.",
                    suggestion="Acknowledge the concern explicitly before responding with facts or next steps.",
                    evidence_turn=u.turn_index, evidence_text=u.text))

    # Pattern 3: rep monologue — a long internal turn with no client turn around it.
    for i, u in enumerate(utts):
        if u.role == ROLE_INTERNAL and len(u.text.split()) > _MONOLOGUE_TOKENS:
            notes.append(CoachingNote(
                kind="rep_monologue",
                observation="A long uninterrupted rep turn may have reduced client participation.",
                suggestion="Break long explanations into smaller pieces and check in with the client.",
                evidence_turn=u.turn_index, evidence_text=u.text[:160]))
    return notes


_LLM_SYSTEM = (
    "You are a sales-coaching assistant. You are given specific transcript moments "
    "(each with a turn index and quote) that were flagged by detectors. Rewrite each "
    "into one short, constructive, specific coaching tip. Do NOT invent moments, do "
    "NOT claim what the client 'would have' done, and keep each tip tied to its turn. "
    "Return ONLY a JSON array of strings, one per input moment, in the same order."
)


def _phrase_with_llm(notes: list[CoachingNote], llm) -> list[CoachingNote]:
    import json
    payload = [{"turn": n.evidence_turn, "kind": n.kind, "quote": n.evidence_text} for n in notes]
    raw = llm.complete(_LLM_SYSTEM, json.dumps(payload), max_tokens=600).strip()
    raw = raw.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    try:
        tips = json.loads(raw)
        if isinstance(tips, list) and len(tips) == len(notes):
            for n, tip in zip(notes, tips):
                n.suggestion = str(tip).strip()
    except (json.JSONDecodeError, ValueError):
        pass  # keep rule-based phrasing on any failure
    return notes


def assert_grounded(report: CoachingReport) -> None:
    """No ungrounded advice: every note must cite a real transcript turn."""
    for n in report.notes:
        if n.evidence_turn is None or n.evidence_turn < 0:
            raise ValueError(f"Ungrounded coaching note without evidence: {n.kind}")


def coach_meeting(meeting: Meeting, scorer: SentimentScorer | None = None, llm=None) -> CoachingReport:
    scorer = scorer or SentimentScorer(backend="lexicon")
    notes = _detect(meeting, scorer)
    method = "rule_based"
    if llm is not None and notes:
        notes = _phrase_with_llm(notes, llm)
        method = "llm_phrased"
    report = CoachingReport(notes=notes, method=method)
    assert_grounded(report)
    return report
