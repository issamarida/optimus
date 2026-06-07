"""Discussion-category classification — the supervised, *trained* component.

Baseline-first: every reported score is compared against a majority-class
``DummyClassifier``. TF-IDF + Logistic Regression is the interpretable workhorse
and the honest thing to try before a transformer. Everything that learns (IDF
stats, coefficients) lives inside the Pipeline, so ``fit`` touches train only.

Transformer/SetFit upgrade: swap the estimator; the ``CategoryClassifier`` API and
the evaluation harness are unchanged.
"""
from __future__ import annotations

from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


def _build_pipeline() -> Pipeline:
    return Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_df=0.9, sublinear_tf=True)),
        ("clf", LogisticRegression(max_iter=2000, class_weight="balanced", C=4.0)),
    ])


class CategoryClassifier:
    """Thin, serialisable wrapper so callers don't reach into the Pipeline."""

    def __init__(self):
        self.pipeline = _build_pipeline()

    def fit(self, texts, labels):
        self.pipeline.fit(texts, labels)
        return self

    def predict(self, texts):
        return self.pipeline.predict(texts)

    def predict_proba(self, texts):
        return self.pipeline.predict_proba(texts)

    @property
    def classes_(self):
        return self.pipeline.named_steps["clf"].classes_


def build_baseline():
    return DummyClassifier(strategy="most_frequent")
