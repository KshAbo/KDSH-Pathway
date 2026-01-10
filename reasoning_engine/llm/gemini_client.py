"""
Gemini LLM client wrapper for the reasoning engine.
"""

import time
from google import genai
from ..config import DRY_RUN

_client = None
_last_call_ts = 0


def _get_client():
    global _client
    if _client is None:
        _client = genai.Client()
    return _client


def call_gemini(prompt: str) -> str:
    """
    Calls Gemini 2.5 Flash and returns raw text.
    Fail-safe: returns __LLM_ERROR__ sentinel on error or rate limit.
    """
    global _last_call_ts

    if DRY_RUN:
        return ""

    now = time.time()

    # Gemini free tier ≈ 5 req/min → ~12s spacing
    if now - _last_call_ts < 12:
        return "__LLM_ERROR__"

    _last_call_ts = now

    try:
        client = _get_client()
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
        )
        return (response.text or "").strip()
    except Exception as e:
        print(f"[GEMINI ERROR] {e}")
        return "__LLM_ERROR__"
