"""meeting_intel — leakage-safe meeting intelligence.

Public surface:
- load_transcripts / Meeting / Utterance   : data + contract
- CategoryClassifier                        : trained discussion-category model
- SentimentScorer / validate_domain_shift   : pretrained sentiment + validation
- summarize_meeting                          : extractive/abstractive summary
- coach_meeting                              : evidence-grounded coaching feedback
- analyze_meeting                            : full per-meeting analysis object
"""
from .schema import Meeting, Utterance
from .data import load_transcripts
from .category import CategoryClassifier
from .sentiment import SentimentScorer, validate_domain_shift
from .summarize import summarize_meeting
from .coaching import coach_meeting
from .pipeline import analyze_meeting

__version__ = "0.2.0"
__all__ = [
    "Meeting", "Utterance", "load_transcripts", "CategoryClassifier",
    "SentimentScorer", "validate_domain_shift", "summarize_meeting",
    "coach_meeting", "analyze_meeting",
]
