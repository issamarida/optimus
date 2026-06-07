from __future__ import annotations

import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from . import config
from .preprocessing import build_preprocessor


def build_category_pipeline(cfg: config.CategoryModelConfig = config.CategoryModelConfig()) -> Pipeline:
    return Pipeline([
        ("features", build_preprocessor(cfg)),
        ("clf", LogisticRegression(max_iter=cfg.max_iter, C=cfg.C, class_weight="balanced")),
    ])


def fit_category(train_df: pd.DataFrame, cfg: config.CategoryModelConfig = config.CategoryModelConfig()) -> Pipeline:
    return build_category_pipeline(cfg).fit(train_df, train_df[config.TARGET_CATEGORY])


def fit_baseline(train_df: pd.DataFrame) -> DummyClassifier:
    return DummyClassifier(strategy="most_frequent").fit(
        train_df[config.NUMERIC_FEATURES], train_df[config.TARGET_CATEGORY]
    )
