"""Download and split the Bitext customer-support dataset.

Run:  python -m src.data
Creates data/train.csv, data/test.csv and data/intent_to_category.json.

Dataset: bitext/Bitext-customer-support-llm-chatbot-training-dataset (Hugging Face)
~26,900 customer messages, 27 intents (e.g. cancel_order, track_refund) grouped in
11 categories (ORDER, REFUND, ...), each with a reference agent response.
"""
import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

HF_NAME = "bitext/Bitext-customer-support-llm-chatbot-training-dataset"
SEED = 42
DATA = Path("data")


def load_raw() -> pd.DataFrame:
    from datasets import load_dataset  # imported here so tests don't need it
    df = load_dataset(HF_NAME, split="train").to_pandas()
    return df.rename(columns={"instruction": "text"})[["text", "intent", "category", "response"]]


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.dropna(subset=["text", "intent"]).copy()
    df["text"] = df["text"].astype(str).str.strip()
    df = df[df["text"].str.len() > 0]
    # Exact duplicate messages would put the same sentence in train AND test (leakage).
    return df.drop_duplicates(subset=["text"]).reset_index(drop=True)


def split(df: pd.DataFrame, train_size: int, test_size: int):
    """Stratified by intent so every class appears in both sets.
    train_size is capped so DistilBERT fine-tuning stays fast on a laptop CPU."""
    rest, test = train_test_split(df, test_size=test_size, stratify=df["intent"], random_state=SEED)
    if train_size < len(rest):
        rest, _ = train_test_split(rest, train_size=train_size, stratify=rest["intent"],
                                   random_state=SEED)
    return rest.reset_index(drop=True), test.reset_index(drop=True)


def main(train_size: int = 4000, test_size: int = 2000, df: pd.DataFrame | None = None,
         out: Path = DATA) -> dict:
    out.mkdir(exist_ok=True)
    df = clean(load_raw() if df is None else df)
    train, test = split(df, train_size, test_size)
    train.to_csv(out / "train.csv", index=False)
    test.to_csv(out / "test.csv", index=False)
    mapping = df.groupby("intent")["category"].first().to_dict()
    (out / "intent_to_category.json").write_text(json.dumps(mapping, indent=2))
    info = {"rows_after_cleaning": len(df), "intents": df["intent"].nunique(),
            "categories": df["category"].nunique(), "train": len(train), "test": len(test)}
    print(json.dumps(info, indent=2))
    return info


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-size", type=int, default=4000)
    ap.add_argument("--test-size", type=int, default=2000)
    a = ap.parse_args()
    main(a.train_size, a.test_size)
