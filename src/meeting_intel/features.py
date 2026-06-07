from __future__ import annotations

import numpy as np
import pandas as pd

from . import config
from .schema import Meeting, ROLE_CLIENT


def _safe_div(num: pd.Series, den: pd.Series) -> np.ndarray:
    num = num.to_numpy(dtype=float)
    den = den.to_numpy(dtype=float)
    return np.divide(num, den, out=np.zeros_like(num), where=den > 0)


def _row(meeting: Meeting) -> dict:
    client = [u for u in meeting.utterances if u.role == ROLE_CLIENT]
    return {
        config.GROUP_COL: meeting.meeting_id,
        config.TEXT_FULL: meeting.full_text,
        config.TEXT_CLIENT: meeting.client_text,
        "n_turns": len(meeting.utterances),
        "n_client_turns": len(client),
        "client_tokens": sum(u.n_tokens for u in client),
        "total_tokens": sum(u.n_tokens for u in meeting.utterances),
        "question_count": sum(1 for u in client if u.is_question),
        config.TARGET_CATEGORY: meeting.category,
        config.TARGET_SENTIMENT: meeting.client_sentiment,
        config.MEETING_OBJ: meeting,
    }


def build_feature_frame(meetings: list[Meeting]) -> pd.DataFrame:
    df = pd.DataFrame([_row(m) for m in meetings])
    for col in (config.TEXT_FULL, config.TEXT_CLIENT):
        df[col] = df[col].str.replace(r"\s+", " ", regex=True).str.strip()
    df["client_turn_share"] = _safe_div(df["n_client_turns"], df["n_turns"])
    df["client_token_share"] = _safe_div(df["client_tokens"], df["total_tokens"])
    df["question_rate"] = _safe_div(df["question_count"], df["n_client_turns"])
    df["mean_turn_tokens"] = _safe_div(df["total_tokens"], df["n_turns"])
    return df
