# Agentic Support-Ticket Triage with a Fine-Tuned Transformer

A LangGraph agent that reads a customer-support ticket, classifies its intent with a
fine-tuned DistilBERT model, routes it to the right team, retrieves grounded answers
(RAG), and drafts a reply with an LLM. Low-confidence tickets go to a human.

```
ticket ─► classify (DistilBERT) ─ confidence ≥ 0.7 ─► retrieve (RAG) ─► draft (LLM) ─► auto_drafted
                                └ confidence < 0.7 ─► human_review ───────────────────► escalated
```

**Covers:** transformers · PyTorch fine-tuning · TF-IDF baseline · agentic AI (LangGraph) ·
RAG with sentence embeddings · human-in-the-loop · evaluation · pytest + GitHub Actions CI.

## Data
[Bitext customer-support dataset](https://huggingface.co/datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset)
on Hugging Face: ~26,900 customer messages, 27 intents in 11 categories, each with a reference
agent answer. Downloaded automatically by `src/data.py`.

## Run
```bash
python -m venv .venv && .venv\Scripts\activate      # Mac/Linux: source .venv/bin/activate
pip install -r requirements.txt
python -m src.data             # download + stratified train/test split
python -m src.baseline         # TF-IDF + logistic regression
python -m src.finetune         # fine-tune DistilBERT (~15-30 min on CPU)
python -m src.evaluate_agent   # evaluate the agent on 300 test tickets
pytest -q
```

### Drafting replies with an LLM (optional)
Default is **Ollama** (free, local): install from https://ollama.com, run
`ollama pull qwen2.5:1.5b`, then `python -m src.evaluate_agent --demo 3`.
Other providers: set `LLM_PROVIDER=anthropic` (+ `ANTHROPIC_API_KEY`) or
`LLM_PROVIDER=openai` (+ `OPENAI_API_KEY`); `LLM_PROVIDER=none` returns the retrieved answer.

## Method
1. **Split**: duplicates removed, then a stratified held-out test set of 2,000 tickets;
   4,000 stratified training tickets (keeps CPU fine-tuning fast).
2. **Baseline**: TF-IDF (1–2-grams) + logistic regression.
3. **Transformer**: `distilbert-base-uncased` fine-tuned in plain PyTorch (AdamW,
   linear warm-up/decay, gradient clipping, max length 64, 2 epochs).
4. **RAG**: training tickets + their reference answers form the knowledge base, embedded with
   `all-MiniLM-L6-v2`; the agent retrieves the 3 closest answers for the predicted intent.
5. **Agent** (LangGraph `StateGraph`): classify → conditional route on confidence →
   retrieve → draft, or human review.
6. **Evaluation**: macro-F1 for both classifiers; for the agent: routing accuracy, intent
   accuracy, escalation rate, precision on auto-handled tickets, retrieval hit@1.

## Results (run on 7 Nov 2026, laptop CPU)
Data: 26,872 tickets → 24,635 after removing 2,237 duplicates; 4,000 train / 2,000 test (stratified).

| Classifier (2,000 test tickets) | Macro-F1 | Accuracy |
|---|---|---|
| TF-IDF + logistic regression (baseline) | 0.985 | 98.5% |
| DistilBERT fine-tuned (2 epochs, ~30 min CPU) | 0.986 | 98.7% |

The transformer matches a strong baseline: this dataset is formulaic, so word patterns already
capture most of the signal.

| Agent (300 test tickets, threshold 0.7) | Value |
|---|---|
| Routing accuracy (right team) | 99.7% |
| Intent accuracy | 99.0% |
| Escalated to human review | 7% |
| Precision on auto-handled tickets | 100% (all 279 correct) |
| Retrieval hit@1 (semantic search alone) | 98.3% |
| Latency | 67 ms / ticket (without LLM) |

All classification errors had low confidence and were escalated, so no wrong reply was sent
automatically.

**LLM drafting** (Qwen 2.5 1.5B via Ollama, 3-ticket demo): replies were grounded in the retrieved
answers, but the small model sometimes ignored details given in the ticket or misused placeholders.
Next steps: a larger model, a prompt that reuses ticket details, and an automatic placeholder check.
