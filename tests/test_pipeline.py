"""Fast tests on synthetic tickets (no downloads, no GPU, no LLM)."""
import json

import numpy as np
import pandas as pd
import pytest

from src import baseline, data
from src.agent import build_agent
from src.llm import LLM, build_prompt
from src.retriever import Retriever

INTENTS = {
    "cancel_order": ("ORDER", ["cancel my order", "i want to cancel the purchase",
                               "please cancel order", "stop my order now"]),
    "track_refund": ("REFUND", ["where is my refund", "track my refund status",
                                "refund not received yet", "check refund progress"]),
    "change_password": ("ACCOUNT", ["reset my password", "change account password",
                                    "forgot my password", "update login password"]),
}


@pytest.fixture
def tickets():
    rng = np.random.default_rng(0)
    rows = []
    for intent, (cat, phrases) in INTENTS.items():
        for i in range(60):
            p = rng.choice(phrases)
            rows.append({"text": f"{p} {rng.choice(['please', 'asap', 'thanks', ''])} #{i}",
                         "intent": intent, "category": cat,
                         "response": f"Answer for {intent} with {{{{Order Number}}}}"})
    return pd.DataFrame(rows)


def test_split_is_stratified_and_disjoint(tickets, tmp_path):
    info = data.main(train_size=90, test_size=60, df=tickets, out=tmp_path)
    train, test = pd.read_csv(tmp_path / "train.csv"), pd.read_csv(tmp_path / "test.csv")
    assert info["intents"] == 3 and len(test) == 60
    assert set(train["intent"]) == set(test["intent"])
    assert not set(train["text"]) & set(test["text"])  # no leakage
    assert json.loads((tmp_path / "intent_to_category.json").read_text())["track_refund"] == "REFUND"


def test_baseline_learns(tickets, tmp_path):
    data.main(train_size=120, test_size=60, df=tickets, out=tmp_path)
    m = baseline.main(data=tmp_path, models=tmp_path / "m", reports=tmp_path / "r")
    assert m["macro_f1"] > 0.9


def test_retriever_finds_same_intent(tickets):
    ret = Retriever(tickets, backend="tfidf")
    hits = ret.search("i need to reset my password", k=3)
    assert hits[0]["intent"] == "change_password"
    filtered = ret.search("anything", k=3, intent="cancel_order")
    assert all(h["intent"] == "cancel_order" for h in filtered)


class FakeClassifier:
    def __init__(self, conf):
        self.conf = conf

    def predict(self, texts):
        return [("track_refund", self.conf) for _ in texts]


def test_agent_routes_by_confidence(tickets):
    ret = Retriever(tickets, backend="tfidf")
    mapping = {k: v[0] for k, v in INTENTS.items()}
    sure = build_agent(FakeClassifier(0.95), ret, LLM("none"), mapping).invoke(
        {"text": "where is my refund"})
    assert sure["status"] == "auto_drafted" and sure["category"] == "REFUND"
    assert "{{Order Number}}" in sure["draft"]
    unsure = build_agent(FakeClassifier(0.40), ret, LLM("none"), mapping).invoke(
        {"text": "hmm something odd"})
    assert unsure["status"] == "escalated" and unsure["draft"] == ""


def test_prompt_contains_ticket_and_references():
    p = build_prompt("my refund?", "track_refund", [{"response": "Refunds take 5 days."}])
    assert "my refund?" in p and "Refunds take 5 days." in p
