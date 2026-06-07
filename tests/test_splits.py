from meeting_intel.data import (grouped_train_test_split, load_transcripts,
                                 meetings_to_frame)


def test_no_meeting_leaks_across_split():
    df = meetings_to_frame(load_transcripts("synthetic", n_meetings=120, seed=1))
    train, test = grouped_train_test_split(df, test_size=0.3, seed=1)
    assert not (set(train["meeting_id"]) & set(test["meeting_id"]))


def test_split_sizes_sum():
    df = meetings_to_frame(load_transcripts("synthetic", n_meetings=100, seed=2))
    train, test = grouped_train_test_split(df, test_size=0.25, seed=2)
    assert len(train) + len(test) == len(df)
