"""Provider-independent LLM adapter (OpenAI-compatible chat endpoint: hosted or local, e.g. Ollama/vLLM).

Configuration via environment: GEOACQUIRER_LLM_URL, GEOACQUIRER_LLM_KEY, GEOACQUIRER_LLM_MODEL.
This code path is NOT exercised by the offline test-suite (no key available in CI); the rule
compiler is the tested default.
"""
from __future__ import annotations

import os
from typing import Callable

import requests


def env_adapter(timeout: float = 60.0) -> Callable[[str], str]:
    url = os.environ.get("GEOACQUIRER_LLM_URL")
    model = os.environ.get("GEOACQUIRER_LLM_MODEL")
    if not url or not model:
        raise RuntimeError("set GEOACQUIRER_LLM_URL and GEOACQUIRER_LLM_MODEL to use an LLM compiler")
    key = os.environ.get("GEOACQUIRER_LLM_KEY", "")

    def call(prompt: str) -> str:
        r = requests.post(
            url.rstrip("/") + "/chat/completions",
            headers={"Authorization": f"Bearer {key}"} if key else {},
            json={"model": model, "messages": [{"role": "user", "content": prompt}], "temperature": 0},
            timeout=timeout,
        )
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]

    return call
