"""Evaluate the agent end to end on held-out tickets.

Run:  python -m src.evaluate_agent              (300 tickets, no LLM: fast)
      python -m src.evaluate_agent --demo 3     (also show 3 replies drafted by your LLM)

Metrics (reports/agent_metrics.json):
  routing_accuracy        predicted category == true category (which team gets the ticket)
  intent_accuracy         predicted intent == true intent
  escalation_rate         share sent to human review (confidence below threshold)
  auto_precision          intent accuracy on the tickets the agent handled automatically
  retrieval_hit_at_1      unfiltered semantic search: the closest knowledge-base entry
                          has the true intent (measures RAG quality on its own)
"""
import argparse
import json
import time
from pathlib import Path

import pandas as pd

from src.agent import build_agent
from src.classifier import load_classifier
from src.llm import LLM
from src.retriever import Retriever


def evaluate(agent, retriever, test: pd.DataFrame, mapping: dict) -> dict:
    rows = []
    t0 = time.time()
    for r in test.itertuples():
        s = agent.invoke({"text": r.text})
        top = retriever.search(r.text, k=1) or [{}]  # independent of the classifier
        rows.append({"intent_ok": s["intent"] == r.intent,
                     "route_ok": s["category"] == mapping.get(r.intent),
                     "escalated": s["status"] == "escalated",
                     "hit1": top[0].get("intent") == r.intent})
    df = pd.DataFrame(rows)
    auto = df[~df["escalated"]]
    return {"tickets": len(df),
            "routing_accuracy": round(df["route_ok"].mean(), 4),
            "intent_accuracy": round(df["intent_ok"].mean(), 4),
            "escalation_rate": round(df["escalated"].mean(), 4),
            "auto_precision": round(auto["intent_ok"].mean(), 4) if len(auto) else None,
            "retrieval_hit_at_1": round(df["hit1"].mean(), 4),
            "seconds_per_ticket": round((time.time() - t0) / len(df), 3)}


def main(n: int = 300, threshold: float = 0.7, demo: int = 0,
         data: Path = Path("data"), reports: Path = Path("reports")) -> dict:
    train, test = pd.read_csv(data / "train.csv"), pd.read_csv(data / "test.csv")
    mapping = json.loads((data / "intent_to_category.json").read_text())
    clf, ret = load_classifier(), Retriever(train)
    print(f"classifier={type(clf).__name__}  retrieval={ret.backend}  threshold={threshold}")

    sample = test.sample(n=min(n, len(test)), random_state=42)
    metrics = evaluate(build_agent(clf, ret, LLM("none"), mapping, threshold), ret, sample, mapping)
    metrics.update({"classifier": type(clf).__name__, "retrieval": ret.backend,
                    "threshold": threshold})
    reports.mkdir(exist_ok=True)
    (reports / "agent_metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))

    if demo:  # show real LLM drafts for a few tickets
        llm = LLM()
        agent = build_agent(clf, ret, llm, mapping, threshold)
        print(f"\n--- Demo with LLM provider={llm.provider} model={llm.model} ---")
        for r in sample.head(demo).itertuples():
            s = agent.invoke({"text": r.text})
            print(f"\nTICKET: {r.text}\nINTENT: {s['intent']} ({s['confidence']:.2f}) "
                  f"-> {s['category']}  [{s['status']}]\nREPLY: {s['draft'] or '(escalated)'}")
    return metrics


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--threshold", type=float, default=0.7)
    ap.add_argument("--demo", type=int, default=0)
    a = ap.parse_args()
    main(a.n, a.threshold, a.demo)
