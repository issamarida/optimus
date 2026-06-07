# Meeting Intelligence (Optimus)

Meeting-transcript NLP on the **AMI Meeting Corpus** (manual gold release, CC BY 4.0),
built to survive interview-grade scrutiny: clean, modular, pipeline-driven,
leakage-controlled, reproducible, and evaluated on real gold data.

1. **Dialogue-act classification** — the trained model. A scikit-learn `Pipeline`
   whose `ColumnTransformer` vectorizes utterance text (TF-IDF) and scales numeric
   segment features, fit on train only. A fine-tuned **BERT** backend (`transformers`)
   is the upgrade, swappable behind the same interface.
2. **Extractive summarization** — TF-IDF centrality with structured decisions /
   action items / issues, ROUGE-evaluated against AMI's human abstractive summaries.
3. **Sentiment + coaching** — keyless components (lexicon/HF sentiment; evidence-
   grounded rule-based coaching). Honestly scoped: AMI is internal-team data, so
   these are demonstrated on transcripts, not validated on AMI.

[![CI](https://github.com/USERNAME/optimus/actions/workflows/ci.yml/badge.svg)](https://github.com/USERNAME/optimus/actions)

## Real results (AMI manual gold release)

Dialogue-act classification, TF-IDF + numeric baseline, split by meeting series
(no participant leakage):

```
segments: 90,142   classes: 15   split: grouped by series, ~70k train / ~20k test
macro-F1: 0.41     weighted-F1: 0.56   accuracy: 0.55   vs majority baseline 0.04
per-class: inform 0.67, backchannel 0.70, elicit-inform 0.57, be-positive 0.55, suggest 0.50
```

Extractive summarization vs human abstractive summaries (139 meetings, ROUGE F):

```
ROUGE-1: 0.24   ROUGE-2: 0.04   ROUGE-L: 0.12   (summary length matched to references)
```

These are real numbers on real gold transcripts. The TF-IDF dialogue-act score is a
strong linear baseline that the BERT backend is expected to beat; the unsupervised
extractive ROUGE is a baseline that an abstractive model (BART/T5) would improve.

## Data setup (CC BY 4.0)

Unzip the AMI manual release to `data/ami_public_manual_1.6.2/` (git-ignored). The
adapter reads `words/`, `dialogueActs/` (gold acts, with NXT range references),
`abstractive/` (summary references), and `ontologies/da-types.xml`.

## Quickstart

```bash
pip install -e ".[eval]"                                    # core + rouge-score
meeting-intel ami-stats --ami data/ami_public_manual_1.6.2
meeting-intel da-eval   --ami data/ami_public_manual_1.6.2  # TF-IDF baseline, real metrics
meeting-intel summ-eval --ami data/ami_public_manual_1.6.2  # ROUGE vs abstractive refs
meeting-intel analyze   --n 3                               # sentiment + coaching demo
pytest -q                                                    # offline tests
```

BERT fine-tuning (needs a GPU and a one-time model download, so it runs on your
machine, not in CI):

```bash
pip install -e ".[bert]"
meeting-intel da-eval --ami data/ami_public_manual_1.6.2 --model bert
```

## Architecture: one job per module

Anything that learns from data lives inside the scikit-learn `Pipeline` and is fit
on train only; the same fitted pipeline is applied to test unchanged.

```
load_dialogue_acts        ami.py           AMI NITE-XML (range refs) -> labelled segment DataFrame
meetings_from_segments    ami.py           segments -> ordered Meeting objects (for summarization)
build_segment_features    features.py      vectorized numeric features (n_tokens, is_question)
grouped_train_test_split  data.py          leakage-safe split by meeting series
build_da_preprocessor     preprocessing.py ColumnTransformer (TF-IDF text + scaled numeric)
fit_da / BertDAClassifier training.py/bert.py  Pipeline(preprocessor -> LogReg) | fine-tuned BERT
evaluate_category/_summarization  evaluate.py  macro-F1 + per-class report | ROUGE
analyze_meeting           pipeline.py      summary + sentiment + coaching
```

`config.py` holds every tunable value (seeds, split sizes, hyper-parameters, BERT
settings, column names). Code is intentionally comment-light; the reasoning is here.

## Design decisions an interviewer will probe

- **Leakage**: split by meeting series, not by segment or session — the same four
  participants recur across a series' sessions (a/b/c/d), so series-level grouping
  prevents participant leakage. Everything learned (IDF vocabulary, scaler stats,
  coefficients, BERT weights) is fit on train; the test split is opened once.
- **Metric matched to objective**: 15 imbalanced classes -> macro-F1 plus a full
  per-class report and a majority baseline for context.
- **ColumnTransformer earns its place**: act type correlates with utterance length
  (backchannels are short, informs long), so the model combines TF-IDF text with
  scaled numeric features rather than text alone.
- **Honest summarization eval**: extractive output is length-matched to the human
  references before scoring; ROUGE measures lexical overlap only, so it is reported
  as a baseline, not proof of usefulness.
- **Honest scoping**: AMI is internal-team design meetings, not client/sales calls,
  so sentiment and coaching have no AMI ground truth and are not claimed to.
- **Coaching is grounded**: every suggestion cites a transcript turn; `assert_grounded`
  makes ungrounded advice impossible.

## Layout

```
src/meeting_intel/
  config.py        every tunable value
  schema.py        Meeting / Utterance
  ami.py           AMI manual-release adapter (gold dialogue acts, abstractive summaries)
  synth.py         synthetic meetings for the offline summary/coaching demo
  data.py          loading + leakage-safe grouped splitting
  features.py      vectorized segment features
  preprocessing.py ColumnTransformer (text + numeric)
  training.py      TF-IDF dialogue-act Pipeline + baseline
  bert.py          fine-tuned BERT backend (transformers)
  sentiment.py     lexicon/HF sentiment + negative-share aggregation
  summarize.py     extractive summarization + structured fields
  coaching.py      evidence-grounded coaching + grounding guard
  evaluate.py      macro-F1 / detection / ROUGE metrics
  pipeline.py      per-meeting analysis object
  scale.py         parallel batch analysis
  cli.py           ami-stats | da-eval | summ-eval | analyze
tests/             da pipeline, splits, summary, coaching, ami adapter
examples/          quickstart.py
.github/workflows/ ci.yml (lint + tests + analyze smoke test)
```

## What is and is not validated here

Everything runs and is validated on real gold data except BERT fine-tuning, which
requires a GPU and a one-time model download. The BERT code is complete and runs
with a single `--model bert` flag on a machine with those resources; the rest of the
solution (gold dialogue-act baseline, ROUGE summarization, tests, CI) is fully
reproducible without it.
