from __future__ import annotations

import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from . import config
from .preprocessing import build_da_preprocessor


def build_da_pipeline(cfg: config.DAModelConfig = config.DAModelConfig()) -> Pipeline:
    return Pipeline([
        ("features", build_da_preprocessor(cfg)),
        ("clf", LogisticRegression(max_iter=cfg.max_iter, C=cfg.C, class_weight="balanced")),
    ])


def fit_da(train_df: pd.DataFrame, cfg: config.DAModelConfig = config.DAModelConfig()) -> Pipeline:
    return build_da_pipeline(cfg).fit(train_df, train_df[config.TARGET_DA])


def fit_baseline(train_df: pd.DataFrame) -> DummyClassifier:
    return DummyClassifier(strategy="most_frequent").fit(
        train_df[config.SEG_NUMERIC], train_df[config.TARGET_DA])
