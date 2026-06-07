from __future__ import annotations

import argparse
import json

from . import config
from .data import grouped_train_test_split, load_transcripts
from .evaluate import choose_threshold_by_cost, evaluate_category, evaluate_detection
from .features import build_feature_frame
from .llm import get_llm
from .scale import analyze_corpus
from .sentiment import SentimentScorer, aligned_negative_share, validate_domain_shift
from .training import fit_baseline, fit_category


def _prepare(split: config.SplitConfig):
    meetings = load_transcripts("synthetic", n_meetings=config.CorpusConfig.n_meetings, seed=split.seed)
    frame = build_feature_frame(meetings)
    train_df, test_df = grouped_train_test_split(frame, split.test_size, split.seed)
    tr_df, val_df = grouped_train_test_split(train_df, split.val_size, split.seed + 1)
    return frame, tr_df, val_df, test_df


def cmd_evaluate(args) -> None:
    split = config.SplitConfig()
    frame, tr_df, val_df, test_df = _prepare(split)
    labels = sorted(frame[config.TARGET_CATEGORY].unique())

    model = fit_category(tr_df)
    baseline = fit_baseline(tr_df)
    model_eval = evaluate_category(test_df[config.TARGET_CATEGORY], model.predict(test_df), labels)
    base_eval = evaluate_category(
        test_df[config.TARGET_CATEGORY], baseline.predict(test_df[config.NUMERIC_FEATURES]), labels)

    scorer = SentimentScorer(backend=args.sentiment)
    detection_cfg = config.DetectionConfig()
    val_truth = (val_df[config.TARGET_SENTIMENT] == config.NEGATIVE).astype(int).to_numpy()
    test_truth = (test_df[config.TARGET_SENTIMENT] == config.NEGATIVE).astype(int).to_numpy()
    threshold = choose_threshold_by_cost(aligned_negative_share(val_df, scorer), val_truth, detection_cfg)
    detection = evaluate_detection(aligned_negative_share(test_df, scorer), test_truth, threshold)

    sample = frame.sample(min(60, len(frame)), random_state=split.seed)
    sent_val = validate_domain_shift(
        scorer, sample[config.TEXT_CLIENT].tolist(), sample[config.TARGET_SENTIMENT].tolist())

    print(json.dumps({
        "split": {"train": len(tr_df), "val": len(val_df), "test": len(test_df)},
        "category": {"model_macro_f1": model_eval["macro_f1"],
                     "baseline_macro_f1": base_eval["macro_f1"],
                     "lift": round(model_eval["macro_f1"] - base_eval["macro_f1"], 3)},
        "negative_interaction_detection": detection,
        "sentiment_domain_validation": sent_val,
    }, indent=2))
    print("\nCategory report (TEST):\n" + model_eval["report"])


def cmd_analyze(args) -> None:
    split = config.SplitConfig()
    _, tr_df, _, test_df = _prepare(split)
    model = fit_category(tr_df)
    scorer = SentimentScorer(backend=args.sentiment)
    meetings = test_df[config.MEETING_OBJ].tolist()[: args.n]
    results = analyze_corpus(meetings, model, scorer=scorer, llm=get_llm(args.llm), n_jobs=1)
    print(json.dumps([r.to_dict() for r in results], indent=2))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="meeting-intel")
    parser.add_argument("--sentiment", default="lexicon", choices=["lexicon", "hf"])
    parser.add_argument("--llm", default="offline")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("evaluate").set_defaults(func=cmd_evaluate)
    analyze = sub.add_parser("analyze")
    analyze.add_argument("--n", type=int, default=3)
    analyze.set_defaults(func=cmd_analyze)
    args = parser.parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
