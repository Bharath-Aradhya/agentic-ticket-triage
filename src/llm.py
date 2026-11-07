"""Pluggable LLM used by the agent to draft replies.

Choose with the LLM_PROVIDER environment variable:
  ollama     free, local (default). Install https://ollama.com, then: ollama pull qwen2.5:1.5b
  anthropic  needs ANTHROPIC_API_KEY
  openai     needs OPENAI_API_KEY
  none       no LLM: returns the retrieved reference answer (fast, used for evaluation)
Model names can be changed with LLM_MODEL.
"""
import os

import requests

SYSTEM = ("You are a customer-support agent. Write a short, polite reply (max 120 words) "
          "to the customer's ticket. Base it ONLY on the reference answers provided. "
          "Keep placeholders like {{Order Number}} unchanged. Do not invent policies.")


def build_prompt(ticket: str, intent: str, references: list[dict]) -> str:
    refs = "\n\n".join(f"Reference {i + 1}:\n{r['response']}" for i, r in enumerate(references))
    return f"Ticket intent: {intent}\n\nCustomer ticket:\n{ticket}\n\n{refs}\n\nYour reply:"


class LLM:
    def __init__(self, provider: str | None = None, model: str | None = None):
        self.provider = (provider or os.getenv("LLM_PROVIDER", "ollama")).lower()
        defaults = {"ollama": "qwen2.5:1.5b", "anthropic": "claude-haiku-5-5",
                    "openai": "gpt-4o-mini", "none": "-"}
        if self.provider not in defaults:
            raise ValueError(f"Unknown LLM_PROVIDER {self.provider!r}")
        self.model = model or os.getenv("LLM_MODEL") or defaults[self.provider]

    def draft(self, ticket: str, intent: str, references: list[dict]) -> str:
        if self.provider == "none" or not references:
            return references[0]["response"] if references else ""
        prompt = build_prompt(ticket, intent, references)
        if self.provider == "ollama":
            r = requests.post("http://localhost:11434/api/chat", timeout=180, json={
                "model": self.model, "stream": False,
                "messages": [{"role": "system", "content": SYSTEM},
                             {"role": "user", "content": prompt}]})
            r.raise_for_status()
            return r.json()["message"]["content"].strip()
        if self.provider == "anthropic":
            r = requests.post("https://api.anthropic.com/v1/messages", timeout=60, headers={
                "x-api-key": os.environ["ANTHROPIC_API_KEY"],
                "anthropic-version": "2023-06-01"}, json={
                "model": self.model, "max_tokens": 300, "system": SYSTEM,
                "messages": [{"role": "user", "content": prompt}]})
            r.raise_for_status()
            return r.json()["content"][0]["text"].strip()
        r = requests.post("https://api.openai.com/v1/chat/completions", timeout=60, headers={
            "Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}"}, json={
            "model": self.model, "max_tokens": 300,
            "messages": [{"role": "system", "content": SYSTEM},
                         {"role": "user", "content": prompt}]})
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"].strip()
