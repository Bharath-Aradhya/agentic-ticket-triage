"""LangGraph agent: classify -> (confident? retrieve -> draft : human review).

    ticket ──► classify ──confidence ≥ threshold──► retrieve (RAG) ──► draft (LLM) ──► done
                   └──────confidence < threshold──► human_review (escalated) ──────► done

Each node is a "tool" step; the conditional edge is the agent's routing decision,
and the human-review branch is the human-in-the-loop safeguard.
"""
from typing import TypedDict

from langgraph.graph import END, StateGraph


class TicketState(TypedDict, total=False):
    text: str
    intent: str
    category: str
    confidence: float
    references: list
    draft: str
    status: str  # "auto_drafted" | "escalated"


def build_agent(classifier, retriever, llm, intent_to_category: dict, threshold: float = 0.7):
    def classify(state: TicketState) -> TicketState:
        intent, conf = classifier.predict([state["text"]])[0]
        return {"intent": intent, "confidence": conf,
                "category": intent_to_category.get(intent, "UNKNOWN")}

    def route(state: TicketState) -> str:
        return "retrieve" if state["confidence"] >= threshold else "human_review"

    def retrieve(state: TicketState) -> TicketState:
        # Metadata filter: only search answers for the predicted intent.
        return {"references": retriever.search(state["text"], k=3, intent=state["intent"])}

    def draft(state: TicketState) -> TicketState:
        return {"draft": llm.draft(state["text"], state["intent"], state["references"]),
                "status": "auto_drafted"}

    def human_review(state: TicketState) -> TicketState:
        # Low confidence: no automatic reply; a person handles it, with suggestions attached.
        return {"references": retriever.search(state["text"], k=3), "draft": "",
                "status": "escalated"}

    g = StateGraph(TicketState)
    g.add_node("classify", classify)
    g.add_node("retrieve", retrieve)
    g.add_node("draft", draft)
    g.add_node("human_review", human_review)
    g.set_entry_point("classify")
    g.add_conditional_edges("classify", route,
                            {"retrieve": "retrieve", "human_review": "human_review"})
    g.add_edge("retrieve", "draft")
    g.add_edge("draft", END)
    g.add_edge("human_review", END)
    return g.compile()
