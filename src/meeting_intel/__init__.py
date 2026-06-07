from .schema import Meeting, Utterance
from .data import load_transcripts, grouped_train_test_split
from .ami import load_dialogue_acts, load_abstractive_references, meetings_from_segments
from .features import build_segment_features
from .training import build_da_pipeline, fit_da, fit_baseline
from .sentiment import SentimentScorer, validate_domain_shift
from .summarize import summarize_meeting
from .coaching import coach_meeting
from .pipeline import analyze_meeting

__version__ = "0.5.0"
__all__ = [
    "Meeting", "Utterance", "load_transcripts", "grouped_train_test_split",
    "load_dialogue_acts", "load_abstractive_references", "meetings_from_segments", "build_segment_features",
    "build_da_pipeline", "fit_da", "fit_baseline", "SentimentScorer",
    "validate_domain_shift", "summarize_meeting", "coach_meeting", "analyze_meeting",
]
