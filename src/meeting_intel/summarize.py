from __future__ import annotations

import json
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

LLM_SYSTEM = (
    "You summarise corporate client meetings for a manager. Be faithful to the "
    "transcript and never invent facts. Respond with ONLY a JSON object with keys "
    "summary (string, <=120 words), decisions (string[]), action_items (string[]), "
    "issues (string[])."
)


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


def _extractive(meeting: Meeting, cfg: config.SummaryConfig) -> MeetingSummary:
    sentences = _sentences(meeting)
    if not sentences:
        return MeetingSummary(summary="(empty transcript)")
    return MeetingSummary(
        summary=" ".join(_top_central(sentences, cfg.max_sentences)),
        decisions=_cue_hits(sentences, DECISION_CUES),
        action_items=_cue_hits(sentences, ACTION_CUES),
        issues=_cue_hits(sentences, ISSUE_CUES),
        method="extractive",
    )


def _abstractive(meeting: Meeting, llm, cfg: config.SummaryConfig) -> MeetingSummary:
    transcript = "\n".join(meeting.transcript_lines())
    raw = llm.complete(LLM_SYSTEM, f"Transcript:\n{transcript}", max_tokens=cfg.llm_max_tokens)
    raw = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return _extractive(meeting, cfg)
    return MeetingSummary(
        summary=str(data.get("summary", "")).strip(),
        decisions=list(data.get("decisions", []))[:8],
        action_items=list(data.get("action_items", []))[:8],
        issues=list(data.get("issues", []))[:8],
        method="abstractive_llm",
    )


def summarize_meeting(meeting: Meeting, llm=None,
                      cfg: config.SummaryConfig = config.SummaryConfig()) -> MeetingSummary:
    return _abstractive(meeting, llm, cfg) if llm is not None else _extractive(meeting, cfg)
