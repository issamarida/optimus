# Meeting Intelligence

Turns corporate/client meeting transcripts into manager-facing analytics:

1. **Discussion-category classification** — a *trained* NLP model (TF-IDF + LogReg baseline → transformer/SetFit upgrade).
2. **Client sentiment** — a *pretrained* model with mandatory domain validation, plus cost-aware negative-interaction detection.
3. **Summarization** — extractive baseline, abstractive LLM upgrade, with structured decisions / action items / issues.
4. **Coaching feedback** — *evidence-grounded*, advisory suggestions tied to specific transcript moments.

Built to survive interview-grade scrutiny: every preprocessing and modelling
decision is justified, leakage-controlled, and reproducible.

[![CI](https://github.com/USERNAME/meeting-intel/actions/workflows/ci.yml/badge.svg)](https://github.com/USERNAME/meeting-intel/actions)

## Quickstart

```bash
pip install -e .            # core only; runs offline with no API key
meeting-intel evaluate      # train category model, report held-out metrics
meeting-intel analyze --n 3 # full per-meeting analysis (category/sentiment/summary/coaching)
pytest -q                   # tests, incl. proofs that leakage controls hold
```

Abstractive summaries + LLM-phrased coaching:

```bash
pip install -e ".[llm]"
export ANTHROPIC_API_KEY=...    # any provider works behind the llm.py abstraction
meeting-intel analyze --n 3 --llm auto
```

## The four problems, modelled differently on purpose

| Task | Strategy | Why |
|---|---|---|
| Discussion category | **Supervised, trained** | Real labels exist (AMI topic/dialogue-act annotations, CC BY 4.0). This is where "trained an NLP model" is literally true. |
| Client sentiment | **Pretrained + domain validation** | Strong checkpoints exist; retraining is wasteful. But they're out-of-domain for B2B meetings, so we *measure* the gap on a hand-labelled sample. |
| Summarization | **Extractive baseline → abstractive LLM** | A deterministic baseline that always runs, plus a production LLM path with structured output. Evaluated with ROUGE against references. |
| Coaching | **Evidence-grounded, advisory** | No ground-truth "good coaching" label exists, so this is decision-support, not a measured model — and every suggestion cites a transcript turn. |

## Design decisions (what an interviewer will probe)

**Problem framing before data.** `schema.py` fixes the unit of observation, keeps
labels off the feature-bearing object, and declares what's available at prediction
time. Optional timing is `None` when absent — that's information, not a zero.

**Leakage is the first-class concern.**
- **Grouped splits by meeting** (`data.py`): utterances in one meeting aren't
  independent, so splitting by utterance leaks meeting-specific quirks into test.
  `GroupShuffleSplit` / `StratifiedGroupKFold` keyed on `meeting_id` prevents it;
  `tests/test_splits.py` asserts no meeting spans train and test.
- **Everything that learns is fit on train only** — IDF stats, coefficients,
  scaling. The test split is opened once, for reporting, and never drives a decision.

**Metrics match the objective** (`evaluate.py`): macro-F1 for the imbalanced
category task (so rare `escalation` isn't ignored); precision / recall / PR-AUC for
negative-interaction detection with a **cost-based threshold** (a missed blow-up is
set 5× worse than a false alarm), chosen on validation and applied to test as-is.

**Coaching is built to be honest, not impressive.** See the module docstring in
`coaching.py`. Suggestions are anchored to deterministically detected moments
(unanswered client question, unacknowledged negative sentiment, rep monologue),
each carrying an `evidence_turn`. We make no counterfactual causal claims, and
`assert_grounded` guarantees nothing free-floats. These are *event detectors* tied
to transcript locations — not the participant "engagement score" that this project
deliberately omits, because that would have been a circular-leakage trap.

**Reproducible & scalable.** Fixed seeds; `scale.py` parallelises per-meeting
analysis with joblib and maps onto Dask/Ray for multi-node compute unchanged, with
partitioning by meeting preserving the leakage boundary.

## Representative demo output (synthetic data)

```
category:        model macro-F1 0.96  vs  baseline 0.05
neg-interaction: precision 0.94  recall 0.83  PR-AUC 0.83
sentiment:       out-of-domain agreement acc 0.70   (the gap is the point)
coaching:        flags unanswered questions / unacknowledged concerns, each cites a turn
```

These validate the **machinery** on seeded synthetic data; they are **not** a
real-world result. Real numbers come from plugging real transcripts into
`load_transcripts` (AMI adapter is the recommended next step).

## Layout

```
src/meeting_intel/
  schema.py       data contract + prediction-time availability
  data.py         loading + leakage-safe grouped splitting
  synth.py        seeded synthetic corpus (with planted coachable moments)
  category.py     trained discussion-category classifier + baseline
  sentiment.py    pretrained sentiment adapter + domain validation
  summarize.py    extractive baseline + abstractive LLM + structured fields
  coaching.py     evidence-grounded coaching, grounding guard
  evaluate.py     category / detection / ROUGE metrics
  llm.py          provider-agnostic LLM client with offline fallback
  pipeline.py     per-meeting analysis object
  scale.py        parallel / multi-node batch analysis
  cli.py          `meeting-intel evaluate | analyze`
tests/            leakage, category-vs-baseline, summary, coaching-grounding
examples/         quickstart.py
.github/workflows ci.yml (lint + tests + CLI smoke test)
```

## Using real data (AMI)

AMI is the recommended backbone: CC BY 4.0 multi-party meeting transcripts with
topic segmentation, dialogue acts, named entities, and abstractive/extractive
**reference summaries** (so summarization gets a real ROUGE evaluation). Write an
adapter that parses AMI into `Meeting` objects, map topic labels to your category
taxonomy, register it in `load_transcripts` — nothing downstream changes. For
sentiment, switch to `SentimentScorer(backend="hf")` and hand-label ~100–300 real
client turns to run `validate_domain_shift`.

## Honest limitations

- Synthetic-data numbers validate plumbing, not method.
- Default sentiment is a lexicon stub; the pretrained backend needs the optional
  deps and a hand-labelled domain sample to be trustworthy.
- Coaching is advisory and unvalidated by construction; treat it as a prompt for
  the rep's own judgement, not an evaluation of their performance.
- Role attribution (client vs internal) is assumed from the source; upstream
  diarisation quality bounds sentiment and coaching accuracy.
```
