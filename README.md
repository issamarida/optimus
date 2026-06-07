# Meeting Intelligence

1. **Discussion-category classification** — a *trained* NLP model (TF-IDF + LogReg baseline → transformer/SetFit upgrade).
2. **Client sentiment** — a *pretrained* model with mandatory domain validation, plus cost-aware negative-interaction detection.
3. **Summarization** — extractive baseline, abstractive LLM upgrade, with structured decisions / action items / issues.
4. **Coaching feedback** — *evidence-grounded*, advisory suggestions tied to specific transcript moments.

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
