from __future__ import annotations

from dataclasses import dataclass

from . import config
from .coaching import CoachingReport, coach_meeting
from .features import build_feature_frame
from .schema import Meeting
from .sentiment import SentimentScorer, aligned_negative_share
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


def analyze_meeting(meeting: Meeting, category_pipeline, scorer: SentimentScorer | None = None,
                    llm=None, cfg: config.DetectionConfig = config.DetectionConfig()) -> MeetingAnalysis:
    scorer = scorer or SentimentScorer()
    frame = build_feature_frame([meeting])
    category = category_pipeline.predict(frame)[0]
    neg_share = float(aligned_negative_share(frame, scorer)[0])
    return MeetingAnalysis(
        meeting_id=meeting.meeting_id,
        category=category,
        client_negative_share=neg_share,
        flag_negative_interaction=neg_share >= cfg.negative_threshold,
        summary=summarize_meeting(meeting, llm=llm),
        coaching=coach_meeting(meeting, scorer=scorer, llm=llm),
    )
