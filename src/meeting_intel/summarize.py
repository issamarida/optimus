from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from . import config
from .schema import Meeting

ACTION_CUES = ("will ", "i'll ", "we'll ", "follow up", "send ", "action", "by next", "by tomorrow")
DECISION_CUES = ("decided", "agreed", "we'll go with", "final", "approved", "confirm")
ISSUE_CUES = ("problem", "issue", "blocked", "concern", "delay", "risk", "outage", "frustrat")
SENTENCE_SPLIT = re.compile(r"(?<=[.?!])\s+")


@dataclass
class MeetingSummary:
    summary: str
    decisions: list[str] = field(default_factory=list)
    action_items: list[str] = field(default_factory=list)
    issues: list[str] = field(default_factory=list)
    method: str = "extractive"

    def to_dict(self) -> dict:
        return {"method": self.method, "summary": self.summary, "decisions": self.decisions,
                "action_items": self.action_items, "issues": self.issues}


def _sentences(meeting: Meeting) -> list[str]:
    out, seen = [], set()
    for utterance in meeting.utterances:
        for sentence in SENTENCE_SPLIT.split(utterance.text.strip()):
            sentence = sentence.strip()
            key = sentence.lower()
            if len(sentence.split()) >= 3 and key not in seen:
                seen.add(key)
                out.append(sentence)
    return out


def _cue_hits(sentences: list[str], cues) -> list[str]:
    return [s for s in sentences if any(cue in s.lower() for cue in cues)][:5]


def _top_central(sentences: list[str], k: int) -> list[str]:
    if len(sentences) <= k:
        return sentences
    matrix = TfidfVectorizer(stop_words="english").fit_transform(sentences)
    centroid = np.asarray(matrix.mean(axis=0))
    scores = cosine_similarity(matrix, centroid).ravel()
    top = sorted(np.argsort(scores)[::-1][:k])
    return [sentences[i] for i in top]


def summarize_meeting(meeting: Meeting, cfg: config.SummaryConfig = config.SummaryConfig()) -> MeetingSummary:
    sentences = _sentences(meeting)
    if not sentences:
        return MeetingSummary(summary="(empty transcript)")
    return MeetingSummary(
        summary=" ".join(_top_central(sentences, cfg.max_sentences)),
        decisions=_cue_hits(sentences, DECISION_CUES),
        action_items=_cue_hits(sentences, ACTION_CUES),
        issues=_cue_hits(sentences, ISSUE_CUES),
    )
