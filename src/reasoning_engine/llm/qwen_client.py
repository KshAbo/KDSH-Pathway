"""
Qwen2.5-14B-Instruct LLM client via Ollama for the reasoning engine.
"""

import requests
from ..config import DRY_RUN

OLLAMA_ENDPOINT = "http://localhost:11434"
OLLAMA_MODEL = "qwen2.5:14b-instruct"


def call_qwen(prompt: str) -> str:
    """
    Calls Qwen2.5-14B-Instruct via Ollama and returns raw text.
    Fail-safe: returns __LLM_ERROR__ sentinel on error.
    """
    if DRY_RUN:
        return ""

    try:
        response = requests.post(
            f"{OLLAMA_ENDPOINT}/api/generate",
            json={
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False,
            },
            timeout=60,
        )

        response.raise_for_status()
        data = response.json()

        if "response" not in data:
            raise RuntimeError(f"Unexpected Ollama response: {data}")

        result = (data["response"] or "").strip()
        return result

    except Exception as e:
        print(f"[QWEN ERROR] {e}")
        return "__LLM_ERROR__"
