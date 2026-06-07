from dataclasses import dataclass

RANDOM_SEED = 7

GROUP_COL = "series"
SEG_TEXT = "text"
TARGET_DA = "da_label"
TARGET_SENTIMENT = "sentiment"
MEETING_OBJ = "_meeting"

SEG_NUMERIC = ["n_tokens", "is_question"]

AMI_ROOT = "data/ami_public_manual_1.6.2"

POSITIVE = "positive"
NEGATIVE = "negative"
NEUTRAL = "neutral"


@dataclass(frozen=True)
class SplitConfig:
    test_size: float = 0.25
    val_size: float = 0.25
    seed: int = RANDOM_SEED


@dataclass(frozen=True)
class DAModelConfig:
    ngram_max: int = 2
    min_df: int = 3
    max_df: float = 0.9
    C: float = 2.0
    max_iter: int = 2000


@dataclass(frozen=True)
class BertConfig:
    model_name: str = "bert-base-uncased"
    max_length: int = 64
    batch_size: int = 32
    epochs: int = 3
    learning_rate: float = 2e-5
    output_dir: str = "artifacts/bert-da"


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


@dataclass(frozen=True)
class CorpusConfig:
    n_meetings: int = 240
    seed: int = RANDOM_SEED
