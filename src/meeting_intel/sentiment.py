from __future__ import annotations

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

from . import config
from .schema import Meeting, ROLE_CLIENT

POS_WORDS = {"great", "excellent", "happy", "perfect", "good", "thank", "exactly",
             "works", "reasonable", "fine"}
NEG_WORDS = {"not", "frustrated", "delays", "delay", "disappointed", "problem",
             "unacceptable", "unhappy", "serious", "outage", "isn't", "can't"}


class SentimentScorer:
    def __init__(self, backend: str = "lexicon", model_name: str | None = None):
        self.backend = backend
        self.model_name = model_name
        self._pipe = None

    def _load(self):
        if self.backend == "hf" and self._pipe is None:
            from transformers import pipeline
            name = self.model_name or "siebert/sentiment-roberta-large-english"
            self._pipe = pipeline("sentiment-analysis", model=name, truncation=True)

    def predict_one(self, text: str) -> str:
        if self.backend == "lexicon":
            toks = set(text.lower().replace("?", " ").replace(".", " ").replace(",", " ").split())
            pos, neg = len(toks & POS_WORDS), len(toks & NEG_WORDS)
            return config.POSITIVE if pos > neg else config.NEGATIVE if neg > pos else config.NEUTRAL
        self._load()
        label = self._pipe(text[:2000])[0]["label"].lower()
        return config.NEGATIVE if "neg" in label else config.POSITIVE if "pos" in label else config.NEUTRAL

    def predict(self, texts) -> list[str]:
        return [self.predict_one(t) for t in texts]


def negative_share_by_meeting(meetings: list[Meeting], scorer: SentimentScorer) -> pd.Series:
    rows = [(m.meeting_id, u.text) for m in meetings for u in m.utterances if u.role == ROLE_CLIENT]
    if not rows:
        return pd.Series(dtype=float)
    df = pd.DataFrame(rows, columns=["meeting_id", "text"])
    df["is_negative"] = pd.Series(scorer.predict(df["text"].tolist())) == config.NEGATIVE
    return df.groupby("meeting_id")["is_negative"].mean()


def validate_domain_shift(scorer: SentimentScorer, texts, gold_labels) -> dict:
    pred = scorer.predict(texts)
    return {
        "n": len(gold_labels),
        "accuracy": round(accuracy_score(gold_labels, pred), 3),
        "macro_f1": round(f1_score(gold_labels, pred, average="macro"), 3),
    }
