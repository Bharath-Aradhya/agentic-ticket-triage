"""Fine-tune DistilBERT (a transformer) to classify ticket intent, in plain PyTorch.

Run:  python -m src.finetune            (about 15-30 min on a laptop CPU)
      python -m src.finetune --epochs 1 (faster)
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report, f1_score
from torch.utils.data import DataLoader, TensorDataset
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          get_linear_schedule_with_warmup)

BASE_MODEL = "distilbert-base-uncased"  # 66M parameters, 6 transformer layers
SEED = 42


def encode(tok, texts, labels, max_len):
    enc = tok(list(texts), truncation=True, padding="max_length", max_length=max_len,
              return_tensors="pt")
    return TensorDataset(enc["input_ids"], enc["attention_mask"], torch.tensor(labels))


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    preds = []
    for ids, mask, _ in loader:
        logits = model(input_ids=ids.to(device), attention_mask=mask.to(device)).logits
        preds.append(logits.argmax(-1).cpu().numpy())
    return np.concatenate(preds)


def main(epochs: int = 2, batch_size: int = 32, lr: float = 5e-5, max_len: int = 64,
         data: Path = Path("data"), out: Path = Path("models/distilbert-intent"),
         reports: Path = Path("reports")) -> dict:
    torch.manual_seed(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train, test = pd.read_csv(data / "train.csv"), pd.read_csv(data / "test.csv")

    labels = sorted(train["intent"].unique())
    label2id = {l: i for i, l in enumerate(labels)}
    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL, num_labels=len(labels), label2id=label2id,
        id2label={i: l for l, i in label2id.items()}).to(device)

    train_dl = DataLoader(encode(tok, train["text"], train["intent"].map(label2id), max_len),
                          batch_size=batch_size, shuffle=True)
    test_dl = DataLoader(encode(tok, test["text"], test["intent"].map(label2id), max_len),
                         batch_size=64)

    # AdamW + linear warm-up/decay: the standard recipe for fine-tuning BERT-style models.
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    steps = epochs * len(train_dl)
    sched = get_linear_schedule_with_warmup(opt, int(0.1 * steps), steps)

    print(f"Fine-tuning {BASE_MODEL} on {len(train)} tickets, {len(labels)} intents, "
          f"{epochs} epochs, device={device}")
    t0 = time.time()
    for ep in range(epochs):
        model.train()
        for step, (ids, mask, y) in enumerate(train_dl, 1):
            loss = model(input_ids=ids.to(device), attention_mask=mask.to(device),
                         labels=y.to(device)).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(), sched.step(), opt.zero_grad()
            if step % 20 == 0 or step == len(train_dl):
                print(f"  epoch {ep + 1}/{epochs}  step {step}/{len(train_dl)}  "
                      f"loss {loss.item():.4f}  {time.time() - t0:.0f}s")

    y_true = test["intent"].map(label2id).to_numpy()
    y_pred = predict(model, test_dl, device)
    metrics = {"macro_f1": round(f1_score(y_true, y_pred, average="macro"), 4),
               "accuracy": round(accuracy_score(y_true, y_pred), 4),
               "train_rows": len(train), "test_rows": len(test), "epochs": epochs,
               "train_minutes": round((time.time() - t0) / 60, 1)}

    out.mkdir(parents=True, exist_ok=True)
    reports.mkdir(exist_ok=True)
    model.save_pretrained(out)
    tok.save_pretrained(out)
    (reports / "distilbert_metrics.json").write_text(json.dumps(metrics, indent=2))
    (reports / "distilbert_per_intent.txt").write_text(
        classification_report(y_true, y_pred, target_names=labels, digits=3))
    print("DistilBERT:", metrics)
    return metrics


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch-size", type=int, default=32)
    a = ap.parse_args()
    main(epochs=a.epochs, batch_size=a.batch_size)
