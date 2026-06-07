from __future__ import annotations

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold

from . import config
from .schema import Meeting


def load_transcripts(source: str = "synthetic", **kwargs) -> list[Meeting]:
    if source == "synthetic":
        from .synth import generate_corpus
        return generate_corpus(**kwargs)
    raise NotImplementedError(f"source={source!r}: add an adapter returning list[Meeting]")


def grouped_train_test_split(df: pd.DataFrame, test_size: float, seed: int):
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_idx, test_idx = next(splitter.split(df, groups=df[config.GROUP_COL]))
    return df.iloc[train_idx].copy(), df.iloc[test_idx].copy()


def grouped_cv(df: pd.DataFrame, y_col: str, n_splits: int = 5, seed: int = config.RANDOM_SEED):
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return list(splitter.split(df, df[y_col], groups=df[config.GROUP_COL]))
