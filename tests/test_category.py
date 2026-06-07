from meeting_intel import build_feature_frame, load_transcripts
from meeting_intel.config import NUMERIC_FEATURES, TARGET_CATEGORY
from meeting_intel.data import grouped_train_test_split
from meeting_intel.evaluate import evaluate_category
from meeting_intel.training import fit_baseline, fit_category


def test_category_beats_majority_baseline():
    df = build_feature_frame(load_transcripts("synthetic", n_meetings=240, seed=7))
    train, test = grouped_train_test_split(df, test_size=0.25, seed=7)
    model = fit_category(train)
    baseline = fit_baseline(train)
    labels = sorted(df[TARGET_CATEGORY].unique())
    m = evaluate_category(test[TARGET_CATEGORY], model.predict(test), labels)
    b = evaluate_category(test[TARGET_CATEGORY], baseline.predict(test[NUMERIC_FEATURES]), labels)
    assert m["macro_f1"] > b["macro_f1"] + 0.2


def test_feature_frame_has_numeric_columns():
    df = build_feature_frame(load_transcripts("synthetic", n_meetings=10, seed=3))
    for col in NUMERIC_FEATURES:
        assert col in df.columns
        assert df[col].between(0, df[col].max() + 1).all()
