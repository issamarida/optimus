"""Batch scoring over many meetings. Meetings are independent, so this is
embarrassingly parallel: joblib fans across local cores, and the same
``analyze_meeting`` unit maps onto Dask/Ray for multi-node compute with no change
to the logic. Partitioning by meeting keeps the leakage boundary intact.
"""
from __future__ import annotations

from joblib import Parallel, delayed

from .pipeline import analyze_meeting


def analyze_corpus(meetings, category_model, scorer=None, llm=None, n_jobs: int = -1):
    # Threads by default so a network-bound LLM backend overlaps I/O; switch to
    # process-based parallelism for CPU-bound local models.
    return Parallel(n_jobs=n_jobs, prefer="threads")(
        delayed(analyze_meeting)(m, category_model, scorer, llm) for m in meetings
    )
