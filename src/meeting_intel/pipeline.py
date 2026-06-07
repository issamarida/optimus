"""Per-meeting orchestration: combine category, sentiment, summary, coaching into
one analysis object. A fitted ``CategoryClassifier`` is required (it must have been
trained on a separate train split); sentiment/summary/coaching are stateless given
their backends.
"""
from __future__ import annotations

from dataclasses import dataclass

from .category import CategoryClassifier
from .coaching import CoachingReport, coach_meeting
from .schema import Meeting, ROLE_CLIENT
from .sentiment import SentimentScorer
from .summarize import MeetingSummary, summarize_meeting


@dataclass
class MeetingAnalysis:
    meeting_id: str
    category: str
    client_negative_share: float
    flag_negative_interaction: bool
    summary: MeetingSummary
    coaching: CoachingReport

    def to_dict(self) -> dict:
        return {
            "meeting_id": self.meeting_id,
            "category": self.category,
            "client_negative_share": round(self.client_negative_share, 3),
            "flag_negative_interaction": self.flag_negative_interaction,
            "summary": self.summary.to_dict(),
            "coaching": self.coaching.to_dict(),
        }


def analyze_meeting(meeting: Meeting, category_model: CategoryClassifier,
                    scorer: SentimentScorer | None = None, llm=None,
                    negative_threshold: float = 0.4) -> MeetingAnalysis:
    scorer = scorer or SentimentScorer(backend="lexicon")
    category = category_model.predict([meeting.full_text])[0]
    client_lines = [u.text for u in meeting.utterances if u.role == ROLE_CLIENT]
    sentiments = scorer.predict(client_lines) if client_lines else []
    neg_share = sum(s == "negative" for s in sentiments) / len(sentiments) if sentiments else 0.0
    return MeetingAnalysis(
        meeting_id=meeting.meeting_id,
        category=category,
        client_negative_share=neg_share,
        flag_negative_interaction=neg_share >= negative_threshold,
        summary=summarize_meeting(meeting, llm=llm),
        coaching=coach_meeting(meeting, scorer=scorer, llm=llm),
    )
