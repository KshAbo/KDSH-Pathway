"""
Ollama HTTP client for LLaMA-3 8B.
"""

import json
import requests
from typing import Optional

from ..config import DRY_RUN, OLLAMA_ENDPOINT, OLLAMA_MODEL


def call_llama(prompt: str) -> str:
    """
    Call LLaMA-3 via Ollama HTTP API.
    
    Args:
        prompt: The prompt to send to the model
        
    Returns:
        The model's response as a string (stripped)
    """
    if DRY_RUN:
        # Return empty string in DRY_RUN mode
        return ""
    
    try:
        response = requests.post(
            f"{OLLAMA_ENDPOINT}/api/generate",
            json={
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False
            },
            timeout=30
        )
        response.raise_for_status()
        
        result = response.json()
        output = result.get("response", "").strip()
        return output
    except requests.exceptions.RequestException as e:
        # On error, return empty string (fail gracefully)
        print(f"Error calling Ollama: {e}")
        return ""
