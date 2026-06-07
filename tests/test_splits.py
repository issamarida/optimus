from meeting_intel import build_feature_frame, load_transcripts
from meeting_intel.config import GROUP_COL
from meeting_intel.data import grouped_train_test_split


def test_no_meeting_leaks_across_split():
    df = build_feature_frame(load_transcripts("synthetic", n_meetings=120, seed=1))
    train, test = grouped_train_test_split(df, test_size=0.3, seed=1)
    assert not (set(train[GROUP_COL]) & set(test[GROUP_COL]))


def test_split_sizes_sum():
    df = build_feature_frame(load_transcripts("synthetic", n_meetings=100, seed=2))
    train, test = grouped_train_test_split(df, test_size=0.25, seed=2)
    assert len(train) + len(test) == len(df)
