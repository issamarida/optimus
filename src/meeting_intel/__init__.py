from .schema import Meeting, Utterance
from .data import load_transcripts, grouped_train_test_split
from .features import build_feature_frame
from .training import build_category_pipeline, fit_category
from .sentiment import SentimentScorer, validate_domain_shift
from .summarize import summarize_meeting
from .coaching import coach_meeting
from .pipeline import analyze_meeting

__version__ = "0.3.0"
__all__ = [
    "Meeting", "Utterance", "load_transcripts", "grouped_train_test_split",
    "build_feature_frame", "build_category_pipeline", "fit_category",
    "SentimentScorer", "validate_domain_shift", "summarize_meeting",
    "coach_meeting", "analyze_meeting",
]
