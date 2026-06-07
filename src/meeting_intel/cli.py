from __future__ import annotations

import argparse
import json

from . import config
from .ami import load_abstractive_references, load_dialogue_acts, meetings_from_segments
from .data import grouped_train_test_split, load_transcripts
from .evaluate import evaluate_category, evaluate_summarization
from .features import build_segment_features
from .scale import analyze_corpus
from .sentiment import SentimentScorer
from .training import fit_baseline, fit_da


def _da_frame(ami_root: str):
    return build_segment_features(load_dialogue_acts(ami_root))


def cmd_ami_stats(args) -> None:
    df = _da_frame(args.ami)
    print(json.dumps({
        "segments": len(df),
        "series": sorted(df[config.GROUP_COL].unique()),
        "labels": df[config.TARGET_DA].value_counts().to_dict(),
        "mean_tokens": round(float(df["n_tokens"].mean()), 2),
    }, indent=2))


def cmd_da_eval(args) -> None:
    split = config.SplitConfig()
    df = _da_frame(args.ami)
    train_df, test_df = grouped_train_test_split(df, split.test_size, split.seed)
    labels = sorted(df[config.TARGET_DA].unique())

    if args.model == "bert":
        from .bert import BertDAClassifier
        model = BertDAClassifier().fit(train_df)
        pred = model.predict(test_df)
    else:
        model = fit_da(train_df)
        pred = model.predict(test_df)

    baseline = fit_baseline(train_df)
    model_eval = evaluate_category(test_df[config.TARGET_DA], pred, labels)
    base_eval = evaluate_category(
        test_df[config.TARGET_DA], baseline.predict(test_df[config.SEG_NUMERIC]), labels)
    print(json.dumps({
        "model": args.model,
        "split": {"train": len(train_df), "test": len(test_df),
                  "train_series": sorted(train_df[config.GROUP_COL].unique()),
                  "test_series": sorted(test_df[config.GROUP_COL].unique())},
        "macro_f1": model_eval["macro_f1"],
        "weighted_f1": model_eval["weighted_f1"],
        "baseline_macro_f1": base_eval["macro_f1"],
    }, indent=2))
    print("\n" + model_eval["report"])


def cmd_analyze(args) -> None:
    meetings = load_transcripts("synthetic", n_meetings=40, seed=config.RANDOM_SEED)[: args.n]
    results = analyze_corpus(meetings, SentimentScorer(backend=args.sentiment), n_jobs=1)
    print(json.dumps([r.to_dict() for r in results], indent=2))



def cmd_summ_eval(args) -> None:
    from .summarize import summarize_meeting
    summ_cfg = config.SummaryConfig(max_sentences=args.max_sentences)
    df = load_dialogue_acts(args.ami)
    refs = load_abstractive_references(args.ami)
    meetings = meetings_from_segments(df)
    predicted = {m.meeting_id: summarize_meeting(m, summ_cfg).summary
                 for m in meetings if m.meeting_id in refs}
    print(json.dumps(evaluate_summarization(predicted, refs), indent=2))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="meeting-intel")
    sub = parser.add_subparsers(dest="cmd", required=True)

    stats = sub.add_parser("ami-stats")
    stats.add_argument("--ami", default=config.AMI_ROOT)
    stats.set_defaults(func=cmd_ami_stats)

    da = sub.add_parser("da-eval")
    da.add_argument("--ami", default=config.AMI_ROOT)
    da.add_argument("--model", default="tfidf", choices=["tfidf", "bert"])
    da.set_defaults(func=cmd_da_eval)

    summ = sub.add_parser("summ-eval")
    summ.add_argument("--ami", default=config.AMI_ROOT)
    summ.add_argument("--max-sentences", type=int, default=30, dest="max_sentences")
    summ.set_defaults(func=cmd_summ_eval)

    analyze = sub.add_parser("analyze")
    analyze.add_argument("--n", type=int, default=3)
    analyze.add_argument("--sentiment", default="lexicon", choices=["lexicon", "hf"])
    analyze.set_defaults(func=cmd_analyze)

    args = parser.parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
