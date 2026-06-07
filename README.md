This version on github does not contain any sensitive data from Optimiza in fact this version has been trained on a meeting-transcript NLP by the AMI Meeting Corpus (manual gold release, CC BY 4.0)

```
segments: 90,142   classes: 15   split: grouped by series, ~70k train / ~20k test
macro-F1: 0.41     weighted-F1: 0.56   accuracy: 0.55   vs majority baseline 0.04
per-class: inform 0.67, backchannel 0.70, elicit-inform 0.57, be-positive 0.55, suggest 0.50
```

```
ROUGE-1: 0.24   ROUGE-2: 0.04   ROUGE-L: 0.12   (summary length matched to references)
```

```bash
pip install -e ".[eval]"                                    # core + rouge-score
meeting-intel ami-stats --ami data/ami_public_manual_1.6.2
meeting-intel da-eval   --ami data/ami_public_manual_1.6.2  # TF-IDF baseline, real metrics
meeting-intel summ-eval --ami data/ami_public_manual_1.6.2  # ROUGE vs abstractive refs
meeting-intel analyze   --n 3                               # sentiment + coaching demo
pytest -q                                                    # offline tests
```

BERT fine-tuning:

```bash
pip install -e ".[bert]"
meeting-intel da-eval --ami data/ami_public_manual_1.6.2 --model bert
```

## Architecture:

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
