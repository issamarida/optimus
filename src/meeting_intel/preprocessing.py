from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler

from . import config


def build_da_preprocessor(cfg: config.DAModelConfig = config.DAModelConfig()) -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("text", TfidfVectorizer(ngram_range=(1, cfg.ngram_max), min_df=cfg.min_df,
                                     max_df=cfg.max_df, sublinear_tf=True), config.SEG_TEXT),
            ("numeric", StandardScaler(), config.SEG_NUMERIC),
        ],
        remainder="drop",
        sparse_threshold=0.3,
    )
