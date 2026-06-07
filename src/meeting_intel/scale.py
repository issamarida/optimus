from __future__ import annotations

from joblib import Parallel, delayed

from .pipeline import analyze_meeting


def analyze_corpus(meetings, category_pipeline, scorer=None, llm=None, n_jobs: int = -1):
    return Parallel(n_jobs=n_jobs, prefer="threads")(
        delayed(analyze_meeting)(m, category_pipeline, scorer, llm) for m in meetings
    )
