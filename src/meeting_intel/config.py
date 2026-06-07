from dataclasses import dataclass

RANDOM_SEED = 7

GROUP_COL = "meeting_id"
TEXT_FULL = "full_text"
TEXT_CLIENT = "client_text"
TARGET_CATEGORY = "category"
TARGET_SENTIMENT = "sentiment"
MEETING_OBJ = "_meeting"

NUMERIC_FEATURES = [
    "client_turn_share",
    "client_token_share",
    "question_rate",
    "mean_turn_tokens",
    "n_turns",
]

POSITIVE = "positive"
NEGATIVE = "negative"
NEUTRAL = "neutral"


@dataclass(frozen=True)
class SplitConfig:
    test_size: float = 0.25
    val_size: float = 0.25
    seed: int = RANDOM_SEED


@dataclass(frozen=True)
class CategoryModelConfig:
    ngram_max: int = 2
    min_df: int = 2
    max_df: float = 0.9
    C: float = 4.0
    max_iter: int = 2000


@dataclass(frozen=True)
class DetectionConfig:
    cost_fn: float = 5.0
    cost_fp: float = 1.0
    negative_threshold: float = 0.4


@dataclass(frozen=True)
class CoachingConfig:
    response_window: int = 2
    monologue_tokens: int = 45


@dataclass(frozen=True)
class SummaryConfig:
    max_sentences: int = 4
    llm_max_tokens: int = 700


@dataclass(frozen=True)
class CorpusConfig:
    n_meetings: int = 240
    seed: int = RANDOM_SEED
