"""TF-IDF + logistic regression baseline: the number DistilBERT has to beat.

Run:  python -m src.baseline
"""
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import Pipeline


def train_baseline(train: pd.DataFrame) -> Pipeline:
    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)),
        ("clf", LogisticRegression(max_iter=2000, C=5.0)),
    ])
    return pipe.fit(train["text"], train["intent"])


def evaluate(pipe: Pipeline, test: pd.DataFrame) -> dict:
    pred = pipe.predict(test["text"])
    return {"macro_f1": round(f1_score(test["intent"], pred, average="macro"), 4),
            "accuracy": round(accuracy_score(test["intent"], pred), 4)}


def main(data: Path = Path("data"), models: Path = Path("models"),
         reports: Path = Path("reports")) -> dict:
    models.mkdir(exist_ok=True)
    reports.mkdir(exist_ok=True)
    train, test = pd.read_csv(data / "train.csv"), pd.read_csv(data / "test.csv")
    pipe = train_baseline(train)
    metrics = evaluate(pipe, test)
    joblib.dump(pipe, models / "baseline.joblib")
    (reports / "baseline_metrics.json").write_text(json.dumps(metrics, indent=2))
    print("TF-IDF + LogReg:", metrics)
    return metrics


if __name__ == "__main__":
    main()
