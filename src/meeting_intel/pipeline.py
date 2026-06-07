from __future__ import annotations

from dataclasses import dataclass

from . import config
from .coaching import CoachingReport, coach_meeting
from .schema import Meeting
from .sentiment import SentimentScorer, negative_share_by_meeting
from .summarize import MeetingSummary, summarize_meeting


@dataclass
class MeetingAnalysis:
    meeting_id: str
    client_negative_share: float
    flag_negative_interaction: bool
    summary: MeetingSummary
    coaching: CoachingReport

    def to_dict(self) -> dict:
        return {
            "meeting_id": self.meeting_id,
            "client_negative_share": round(self.client_negative_share, 3),
            "flag_negative_interaction": self.flag_negative_interaction,
            "summary": self.summary.to_dict(),
            "coaching": self.coaching.to_dict(),
        }


def analyze_meeting(meeting: Meeting, scorer: SentimentScorer | None = None,
                    cfg: config.DetectionConfig = config.DetectionConfig()) -> MeetingAnalysis:
    scorer = scorer or SentimentScorer()
    shares = negative_share_by_meeting([meeting], scorer)
    neg_share = float(shares.get(meeting.meeting_id, 0.0))
    return MeetingAnalysis(
        meeting_id=meeting.meeting_id,
        client_negative_share=neg_share,
        flag_negative_interaction=neg_share >= cfg.negative_threshold,
        summary=summarize_meeting(meeting),
        coaching=coach_meeting(meeting, scorer=scorer),
    )
