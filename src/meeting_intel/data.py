"""Loading + leakage-safe splitting.

The central leakage control: split by MEETING, never by utterance. Utterances in
one meeting share topic, participants, and tone, so letting them straddle the
train/test boundary inflates scores. ``GroupShuffleSplit`` keyed on ``meeting_id``
makes that impossible.
"""
from __future__ import annotations

from typing import Iterable

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold

from .schema import Meeting


def load_transcripts(source: str = "synthetic", **kwargs) -> list[Meeting]:
    """Single entry point. Add adapters here to plug in real data (e.g. AMI):
    parse into ``Meeting`` objects with labels on the object, and the rest of the
    pipeline is unchanged."""
    if source == "synthetic":
        from .synth import generate_corpus
        return generate_corpus(**kwargs)
    raise NotImplementedError(
        f"source={source!r}: write an adapter returning list[Meeting]."
    )


def meetings_to_frame(meetings: Iterable[Meeting]) -> pd.DataFrame:
    rows = [{
        "meeting_id": m.meeting_id,
        "full_text": m.full_text,
        "_category": m.category,
        "_sentiment": m.client_sentiment,
        "_meeting_obj": m,
    } for m in meetings]
    return pd.DataFrame(rows)


def grouped_train_test_split(df: pd.DataFrame, test_size: float = 0.25, seed: int = 7):
    """Hold out whole meetings. The test split is a protected resource: touched
    once, at the end, never used to make a decision."""
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    tr, te = next(gss.split(df, groups=df["meeting_id"]))
    return df.iloc[tr].copy(), df.iloc[te].copy()


def grouped_cv(df: pd.DataFrame, y_col: str, n_splits: int = 5, seed: int = 7):
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return list(sgkf.split(df, df[y_col], groups=df["meeting_id"]))
