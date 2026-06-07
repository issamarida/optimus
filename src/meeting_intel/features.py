from __future__ import annotations

import pandas as pd

from . import config


def build_segment_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["n_tokens"] = out[config.SEG_TEXT].str.split().str.len().fillna(0).astype(float)
    out["is_question"] = out[config.SEG_TEXT].str.contains(r"\?", regex=True, na=False).astype(float)
    return out
