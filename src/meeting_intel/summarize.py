"""Meeting summarization.

Two paths, same output shape:

- Extractive (default, deterministic, no network): rank transcript sentences by
  cosine similarity to the meeting's TF-IDF centroid (a lightweight, well-known
  baseline) and return the top few in original order. Plus cheap structured
  extraction of likely action items / decisions / issues via cue phrases.
- Abstractive (production): an LLM produces a structured summary. Trigger by
  passing an ``llm`` (see ``meeting_intel.llm.get_llm``).

Evaluation: against AMI's reference abstractive summaries with ROUGE
(``rouge-score`` package). ROUGE only measures lexical overlap, so the README is
explicit about its limits; human review remains the gold standard for usefulness.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .schema import Meeting

_ACTION_CUES = ("will ", "i'll ", "we'll ", "follow up", "send ", "action", "by next", "by tomorrow")
_DECISION_CUES = ("decided", "agreed", "we'll go with", "final", "approved", "confirm")
_ISSUE_CUES = ("problem", "issue", "blocked", "concern", "delay", "risk", "outage", "frustrat")


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
    for u in meeting.utterances:
        for s in re.split(r"(?<=[.?!])\s+", u.text.strip()):
            s = s.strip()
            key = s.lower()
            if len(s.split()) >= 3 and key not in seen:
                seen.add(key)
                out.append(s)
    return out


def _cue_hits(sentences: list[str], cues) -> list[str]:
    hits = [s for s in sentences if any(c in s.lower() for c in cues)]
    seen, uniq = set(), []
    for h in hits:
        if h.lower() not in seen:
            seen.add(h.lower())
            uniq.append(h)
    return uniq[:5]


def _extractive(meeting: Meeting, max_sentences: int = 4) -> MeetingSummary:
    sents = _sentences(meeting)
    if not sents:
        return MeetingSummary(summary="(empty transcript)")
    if len(sents) <= max_sentences:
        chosen = sents
    else:
        tfidf = TfidfVectorizer(stop_words="english").fit_transform(sents)
        centroid = np.asarray(tfidf.mean(axis=0))
        scores = cosine_similarity(tfidf, centroid).ravel()
        top = sorted(sorted(range(len(sents)), key=lambda i: -scores[i])[:max_sentences])
        chosen = [sents[i] for i in top]
    return MeetingSummary(
        summary=" ".join(chosen),
        decisions=_cue_hits(sents, _DECISION_CUES),
        action_items=_cue_hits(sents, _ACTION_CUES),
        issues=_cue_hits(sents, _ISSUE_CUES),
        method="extractive",
    )


_LLM_SYSTEM = (
    "You summarise corporate client meetings for a manager. Be faithful to the "
    "transcript and never invent facts. Respond with ONLY a JSON object with keys "
    "summary (string, <=120 words), decisions (string[]), action_items (string[]), "
    "issues (string[]). No prose outside the JSON."
)


def _abstractive(meeting: Meeting, llm) -> MeetingSummary:
    transcript = "\n".join(meeting.transcript_lines())
    raw = llm.complete(_LLM_SYSTEM, f"Transcript:\n{transcript}", max_tokens=700)
    raw = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return _extractive(meeting)  # graceful fallback if the model misbehaves
    return MeetingSummary(
        summary=str(data.get("summary", "")).strip(),
        decisions=list(data.get("decisions", []))[:8],
        action_items=list(data.get("action_items", []))[:8],
        issues=list(data.get("issues", []))[:8],
        method="abstractive_llm",
    )


def summarize_meeting(meeting: Meeting, llm=None, max_sentences: int = 4) -> MeetingSummary:
    if llm is not None:
        return _abstractive(meeting, llm)
    return _extractive(meeting, max_sentences=max_sentences)
