from meeting_intel.category import CategoryClassifier, build_baseline
from meeting_intel.data import (grouped_train_test_split, load_transcripts,
                                 meetings_to_frame)
from meeting_intel.evaluate import evaluate_category


def test_category_beats_majority_baseline():
    df = meetings_to_frame(load_transcripts("synthetic", n_meetings=240, seed=7))
    train, test = grouped_train_test_split(df, test_size=0.25, seed=7)
    model = CategoryClassifier().fit(train["full_text"], train["_category"])
    base = build_baseline().fit(train["full_text"], train["_category"])
    labels = sorted(df["_category"].unique())
    m = evaluate_category(test["_category"], model.predict(test["full_text"]), labels)
    b = evaluate_category(test["_category"], base.predict(test["full_text"]), labels)
    assert m["macro_f1"] > b["macro_f1"] + 0.2
