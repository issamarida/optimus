"""Client sentiment — pretrained, with mandatory domain validation.

We don't retrain sentiment from scratch (strong checkpoints exist) but we never
trust an out-of-domain model blindly. Two parts:

1. ``SentimentScorer`` with a swappable backend. Default ``lexicon`` runs with no
   downloads; ``hf`` wraps a Hugging Face model (e.g.
   ``siebert/sentiment-roberta-large-english``), which is out-of-domain for B2B
   meetings — exactly why part 2 exists.
2. ``validate_domain_shift`` measures agreement against a hand-labelled sample of
   REAL client turns. That number is the honest claim, not an assumption.
"""
from __future__ import annotations

from sklearn.metrics import accuracy_score, f1_score

_POS_WORDS = {"great", "excellent", "happy", "perfect", "good", "thank", "exactly",
              "works", "reasonable", "fine"}
_NEG_WORDS = {"not", "frustrated", "delays", "delay", "disappointed", "problem",
              "unacceptable", "unhappy", "serious", "outage", "isn't", "can't"}


class SentimentScorer:
    def __init__(self, backend: str = "lexicon", model_name: str | None = None):
        self.backend = backend
        self.model_name = model_name
        self._pipe = None

    def _load(self):
        if self.backend == "hf" and self._pipe is None:
            from transformers import pipeline  # optional dep
            name = self.model_name or "siebert/sentiment-roberta-large-english"
            self._pipe = pipeline("sentiment-analysis", model=name, truncation=True)

    def predict_one(self, text: str) -> str:
        if self.backend == "lexicon":
            toks = set(text.lower().replace("?", " ").replace(".", " ").replace(",", " ").split())
            pos, neg = len(toks & _POS_WORDS), len(toks & _NEG_WORDS)
            return "positive" if pos > neg else "negative" if neg > pos else "neutral"
        self._load()
        label = self._pipe(text[:2000])[0]["label"].lower()
        return "negative" if "neg" in label else "positive" if "pos" in label else "neutral"

    def predict(self, texts):
        return [self.predict_one(t) for t in texts]


def validate_domain_shift(scorer: SentimentScorer, texts, gold_labels) -> dict:
    pred = scorer.predict(texts)
    return {
        "n": len(gold_labels),
        "accuracy": round(accuracy_score(gold_labels, pred), 3),
        "macro_f1": round(f1_score(gold_labels, pred, average="macro"), 3),
    }
