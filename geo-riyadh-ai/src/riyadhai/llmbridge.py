"""Budgeted sub-model calls for analysis code (the recursive-language-model step).

Inside the sandbox the code can call `llm_query(prompt)` or `llm_map(items, instruction)`. The sandbox itself has no network and no keys: each call is
a file handshake serviced by THIS bridge in the parent process, which enforces:
  * a call budget, a per-call size limit and a total-size limit (the budget cannot be raised from inside the sandbox)
  * depth 1: a sub-call is a plain completion with no tools, so it cannot call further models or run code
  * a response cache (same prompt, same answer, no second charge)
  * a full audit log (prompt and answer previews, sizes, seconds) shown to the user
Use it for what code cannot do (reading fuzzy Arabic/English names, classifying free text, summarising); never for counting, distances or joins.
Whatever a sub-call returns is a model JUDGEMENT, and the answer says so.
"""
from __future__ import annotations

import hashlib
import time
from typing import Callable, Optional

SUB_SYSTEM = ("You are a precise helper inside a data-analysis pipeline. Do exactly what the instruction says using ONLY the text provided. "
              "If something is not determinable from the input, answer null/unknown for it; never guess or invent facts. "
              "When asked for JSON, output only valid JSON and nothing else.")


class BudgetExceeded(RuntimeError):
    pass


class LLMBridge:
    def __init__(self, llm: Callable[[str, list], str], max_calls: int = 30, max_chars_per_call: int = 14000, max_total_chars: int = 250000):
        self.llm, self.max_calls, self.max_chars, self.max_total = llm, max_calls, max_chars_per_call, max_total_chars
        self.calls: list[dict] = []
        self.cache: dict[str, str] = {}
        self.total_chars = 0

    def __call__(self, prompt: str, opts: Optional[dict] = None) -> str:
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("llm_query needs a non-empty text prompt")
        if len(prompt) > self.max_chars:
            raise BudgetExceeded(f"prompt is {len(prompt):,} characters; the limit is {self.max_chars:,}. Split the input into smaller batches")
        key = hashlib.sha1(prompt.encode("utf-8")).hexdigest()
        if key in self.cache:
            self.calls.append({"cached": True, "prompt": prompt[:160], "answer": self.cache[key][:160], "chars": len(prompt), "seconds": 0.0})
            return self.cache[key]
        if len(self.calls) - sum(1 for c in self.calls if c.get("cached")) >= self.max_calls:
            raise BudgetExceeded(f"the model-call budget ({self.max_calls}) is used up; use fewer, larger batches")
        if self.total_chars + len(prompt) > self.max_total:
            raise BudgetExceeded("the total text budget for model calls is used up")
        t0 = time.monotonic()
        text = self.llm(SUB_SYSTEM, [{"role": "user", "content": prompt}])
        self.total_chars += len(prompt)
        self.cache[key] = text
        self.calls.append({"cached": False, "prompt": prompt[:160], "answer": text[:160], "chars": len(prompt), "seconds": round(time.monotonic() - t0, 2)})
        return text

    def summary(self) -> dict:
        paid = [c for c in self.calls if not c.get("cached")]
        return {"calls": len(paid), "cached": len(self.calls) - len(paid), "budget_calls": self.max_calls, "chars_sent": self.total_chars, "log": self.calls[-40:]}
