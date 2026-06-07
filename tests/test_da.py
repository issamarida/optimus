import pandas as pd

from meeting_intel.config import SEG_TEXT, TARGET_DA, SEG_NUMERIC, GROUP_COL
from meeting_intel.data import grouped_train_test_split
from meeting_intel.features import build_segment_features
from meeting_intel.training import fit_da

_ROWS = [
    {SEG_TEXT: "yeah okay right", TARGET_DA: "bck", GROUP_COL: "A"},
    {SEG_TEXT: "i think we should use a smaller chip", TARGET_DA: "sug", GROUP_COL: "A"},
    {SEG_TEXT: "the budget is fixed at twelve euros", TARGET_DA: "inf", GROUP_COL: "B"},
    {SEG_TEXT: "what do you think about the colour", TARGET_DA: "el.inf", GROUP_COL: "B"},
] * 25


def _frame():
    return build_segment_features(pd.DataFrame(_ROWS))


def test_segment_features_are_numeric():
    df = _frame()
    for col in SEG_NUMERIC:
        assert col in df.columns
        assert df[col].notna().all()


def test_da_pipeline_fits_and_predicts():
    df = _frame()
    train, test = grouped_train_test_split(df, test_size=0.5, seed=1)
    model = fit_da(train)
    preds = model.predict(test)
    assert len(preds) == len(test)
    assert set(preds).issubset(set(df[TARGET_DA]))


def test_grouped_split_has_no_series_overlap():
    df = _frame()
    train, test = grouped_train_test_split(df, test_size=0.5, seed=1)
    assert not (set(train[GROUP_COL]) & set(test[GROUP_COL]))
