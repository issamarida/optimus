"""Command-line interface.

    meeting-intel evaluate            # train category model, report held-out metrics
    meeting-intel analyze --n 3       # full analysis of a few meetings (JSON)

Both default to the synthetic corpus and the offline backends so they run with no
network or API key. Set ANTHROPIC_API_KEY and --llm anthropic for abstractive
summaries and LLM-phrased coaching.
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from .category import CategoryClassifier, build_baseline
from .data import grouped_train_test_split, load_transcripts, meetings_to_frame
from .evaluate import (choose_threshold_by_cost, evaluate_category,
                       evaluate_detection)
from .llm import get_llm
from .scale import analyze_corpus
from .sentiment import SentimentScorer, validate_domain_shift

SEED = 7


def _train_and_split():
    meetings = load_transcripts("synthetic", n_meetings=240, seed=SEED)
    df = meetings_to_frame(meetings)
    train_df, test_df = grouped_train_test_split(df, test_size=0.25, seed=SEED)
    tr_df, val_df = grouped_train_test_split(train_df, test_size=0.25, seed=SEED + 1)
    model = CategoryClassifier().fit(tr_df["full_text"], tr_df["_category"])
    return meetings, df, tr_df, val_df, test_df, model


def cmd_evaluate(args) -> None:
    meetings, df, tr_df, val_df, test_df, model = _train_and_split()
    labels = sorted(df["_category"].unique())

    cat = evaluate_category(test_df["_category"], model.predict(test_df["full_text"]), labels)
    base = build_baseline().fit(tr_df["full_text"], tr_df["_category"])
    base_eval = evaluate_category(test_df["_category"], base.predict(test_df["full_text"]), labels)

    scorer = SentimentScorer(backend=args.sentiment)

    def neg_share(objs):
        out = []
        for m in objs:
            lines = [u.text for u in m.utterances if u.role == "client"]
            s = scorer.predict(lines) if lines else []
            out.append(sum(x == "negative" for x in s) / len(s) if s else 0.0)
        return np.asarray(out)

    val_scores = neg_share(val_df["_meeting_obj"])
    threshold = choose_threshold_by_cost(
        val_scores, (val_df["_sentiment"] == "negative").astype(int).values, cost_fn=5.0, cost_fp=1.0)
    detection = evaluate_detection(
        neg_share(test_df["_meeting_obj"]),
        (test_df["_sentiment"] == "negative").astype(int).values, threshold)

    sample = df.sample(min(60, len(df)), random_state=SEED)
    sent_val = validate_domain_shift(
        scorer, [m.client_text for m in sample["_meeting_obj"]], list(sample["_sentiment"]))

    print(json.dumps({
        "split": {"train": len(tr_df), "val": len(val_df), "test": len(test_df)},
        "category": {"model_macro_f1": cat["macro_f1"],
                     "baseline_macro_f1": base_eval["macro_f1"],
                     "lift": round(cat["macro_f1"] - base_eval["macro_f1"], 3)},
        "negative_interaction_detection": detection,
        "sentiment_domain_validation": sent_val,
    }, indent=2))
    print("\nCategory report (TEST):\n" + cat["report"])


def cmd_analyze(args) -> None:
    _, _, _, _, test_df, model = _train_and_split()
    scorer = SentimentScorer(backend=args.sentiment)
    llm = get_llm(args.llm)
    objs = list(test_df["_meeting_obj"])[: args.n]
    results = analyze_corpus(objs, model, scorer=scorer, llm=llm, n_jobs=1)
    print(json.dumps([r.to_dict() for r in results], indent=2))


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="meeting-intel")
    p.add_argument("--sentiment", default="lexicon", choices=["lexicon", "hf"])
    p.add_argument("--llm", default="offline", help="offline | auto | anthropic")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("evaluate").set_defaults(func=cmd_evaluate)
    a = sub.add_parser("analyze")
    a.add_argument("--n", type=int, default=3)
    a.set_defaults(func=cmd_analyze)
    args = p.parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
