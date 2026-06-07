from __future__ import annotations

import numpy as np
import pandas as pd

from . import config


class BertDAClassifier:
    def __init__(self, cfg: config.BertConfig = config.BertConfig()):
        self.cfg = cfg
        self.labels: list[str] = []
        self._model = None
        self._tokenizer = None

    def _encode(self, texts):
        return self._tokenizer(list(texts), truncation=True, max_length=self.cfg.max_length,
                               padding=True, return_tensors="pt")

    def fit(self, train_df: pd.DataFrame, eval_df: pd.DataFrame | None = None):
        import torch
        from torch.utils.data import Dataset
        from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                                   Trainer, TrainingArguments)

        self.labels = sorted(train_df[config.TARGET_DA].unique())
        label2id = {label: i for i, label in enumerate(self.labels)}
        self._tokenizer = AutoTokenizer.from_pretrained(self.cfg.model_name)

        class _DS(Dataset):
            def __init__(self, frame, tokenizer, cfg):
                self.enc = tokenizer(frame[config.SEG_TEXT].tolist(), truncation=True,
                                     max_length=cfg.max_length, padding=True)
                self.y = [label2id[v] for v in frame[config.TARGET_DA]]

            def __len__(self):
                return len(self.y)

            def __getitem__(self, idx):
                item = {k: torch.tensor(v[idx]) for k, v in self.enc.items()}
                item["labels"] = torch.tensor(self.y[idx])
                return item

        self._model = AutoModelForSequenceClassification.from_pretrained(
            self.cfg.model_name, num_labels=len(self.labels),
            id2label={i: lbl for lbl, i in label2id.items()}, label2id=label2id)

        args = TrainingArguments(
            output_dir=self.cfg.output_dir, num_train_epochs=self.cfg.epochs,
            per_device_train_batch_size=self.cfg.batch_size,
            per_device_eval_batch_size=self.cfg.batch_size,
            learning_rate=self.cfg.learning_rate, eval_strategy="epoch" if eval_df is not None else "no",
            save_strategy="no", logging_steps=50, report_to=[])
        trainer = Trainer(
            model=self._model, args=args, train_dataset=_DS(train_df, self._tokenizer, self.cfg),
            eval_dataset=_DS(eval_df, self._tokenizer, self.cfg) if eval_df is not None else None)
        trainer.train()
        return self

    def predict(self, df: pd.DataFrame):
        import torch
        self._model.eval()
        texts = df[config.SEG_TEXT].tolist()
        out = []
        for start in range(0, len(texts), self.cfg.batch_size):
            batch = self._encode(texts[start:start + self.cfg.batch_size])
            with torch.no_grad():
                logits = self._model(**batch).logits.cpu().numpy()
            out.extend(np.asarray(self.labels)[logits.argmax(axis=1)])
        return np.asarray(out)

    def save(self, path: str | None = None):
        path = path or self.cfg.output_dir
        self._model.save_pretrained(path)
        self._tokenizer.save_pretrained(path)
