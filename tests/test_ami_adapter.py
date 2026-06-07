from meeting_intel.ami import _series, load_abstractive_references


def test_series_extraction():
    assert _series("IS1009b") == "IS1009"
    assert _series("TS3004d") == "TS3004"


def test_extractive_references_missing_dir_is_empty():
    assert load_abstractive_references("/nonexistent/path") == {}
