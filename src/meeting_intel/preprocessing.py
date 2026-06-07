from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler

from . import config


def _tfidf(cfg: config.CategoryModelConfig) -> TfidfVectorizer:
    return TfidfVectorizer(
        ngram_range=(1, cfg.ngram_max),
        min_df=cfg.min_df,
        max_df=cfg.max_df,
        sublinear_tf=True,
    )


def build_preprocessor(cfg: config.CategoryModelConfig = config.CategoryModelConfig()) -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("text_full", _tfidf(cfg), config.TEXT_FULL),
            ("text_client", _tfidf(cfg), config.TEXT_CLIENT),
            ("numeric", StandardScaler(), config.NUMERIC_FEATURES),
        ],
        remainder="drop",
        sparse_threshold=0.3,
    )
