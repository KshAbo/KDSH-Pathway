"""
Ollama HTTP client for LLaMA-3 8B.
"""

import requests
from reasoning_engine.config import DRY_RUN, OLLAMA_ENDPOINT, OLLAMA_MODEL


def call_llama(prompt: str) -> str:
    """
    Call LLaMA-3 via Ollama HTTP API.

    Fails loudly if DRY_RUN is False and Ollama is unreachable.
    """

    if DRY_RUN:
        raise RuntimeError("call_llama() was called while DRY_RUN=True")

    try:
        response = requests.post(
            f"{OLLAMA_ENDPOINT}/api/generate",
            json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False},
            timeout=60,
        )

        response.raise_for_status()
        data = response.json()

        if "response" not in data:
            raise RuntimeError(f"Unexpected Ollama response: {data}")

        return data["response"].strip()

    except Exception as e:
        print("\n[FATAL] Ollama call failed.")
        print("Reason:", e)
        print("\nMake sure Ollama is running:")
        print("  ollama run llama3:8b\n")
        raise
