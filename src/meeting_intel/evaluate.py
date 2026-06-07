"""Evaluation — metrics chosen to match each objective.

- Category (imbalanced multiclass): macro-F1 + confusion + majority baseline.
- Negative-interaction detection: precision / recall / PR-AUC with a cost-based
  threshold picked on validation (a missed blow-up costs more than a false alarm).
- Sentiment: agreement on a hand-labelled domain sample (see sentiment module).
- Summarization: ROUGE against reference summaries when available. ROUGE measures
  lexical overlap only; treat it as a proxy, not proof of usefulness.

The test set is opened exactly once and reported as-is.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (average_precision_score, classification_report,
                             confusion_matrix, f1_score)


def evaluate_category(y_true, y_pred, labels=None) -> dict:
    return {
        "macro_f1": round(f1_score(y_true, y_pred, average="macro"), 3),
        "weighted_f1": round(f1_score(y_true, y_pred, average="weighted"), 3),
        "report": classification_report(y_true, y_pred, labels=labels, zero_division=0),
        "confusion": confusion_matrix(y_true, y_pred, labels=labels),
        "labels": labels,
    }


def choose_threshold_by_cost(scores, y_true, cost_fn: float = 5.0, cost_fp: float = 1.0) -> float:
    scores, y = np.asarray(scores), np.asarray(y_true).astype(int)
    best_t, best_cost = 0.5, float("inf")
    for t in np.linspace(0.05, 0.95, 91):
        pred = (scores >= t).astype(int)
        cost = cost_fn * int(((pred == 0) & (y == 1)).sum()) + cost_fp * int(((pred == 1) & (y == 0)).sum())
        if cost < best_cost:
            best_cost, best_t = cost, round(float(t), 3)
    return best_t


def evaluate_detection(scores, y_true, threshold) -> dict:
    scores, y = np.asarray(scores), np.asarray(y_true).astype(int)
    pred = (scores >= threshold).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    return {
        "threshold": threshold,
        "precision": round(tp / (tp + fp), 3) if (tp + fp) else 0.0,
        "recall": round(tp / (tp + fn), 3) if (tp + fn) else 0.0,
        "pr_auc": round(average_precision_score(y, scores), 3) if y.sum() else None,
        "tp": tp, "fp": fp, "fn": fn,
    }


def evaluate_summaries(predicted: list[str], references: list[str]) -> dict:
    """ROUGE-1/2/L F-measures. Requires `pip install rouge-score`."""
    try:
        from rouge_score import rouge_scorer
    except ImportError:
        return {"error": "install rouge-score to evaluate summaries"}
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    agg = {k: [] for k in ("rouge1", "rouge2", "rougeL")}
    for pred, ref in zip(predicted, references):
        s = scorer.score(ref, pred)
        for k in agg:
            agg[k].append(s[k].fmeasure)
    return {k: round(float(np.mean(v)), 3) for k, v in agg.items()}
