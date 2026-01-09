"""
Contradiction detection logic.
"""

import json
import os
from typing import List

from ..config import DRY_RUN
from ..llm.llama_client import call_llama
from ..llm.prompts import get_contradiction_prompt

# Cache file path
CACHE_DIR = os.path.join(os.path.dirname(__file__), "..", "cache")
CACHE_FILE = os.path.join(CACHE_DIR, "contradiction_cache.json")


def _load_cache() -> dict:
    """Load contradiction cache from disk."""
    if not os.path.exists(CACHE_FILE):
        return {}

    try:
        with open(CACHE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return {}


def _save_cache(cache: dict):
    """Save contradiction cache to disk."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    try:
        with open(CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(cache, f, indent=2, ensure_ascii=False)
    except IOError:
        pass  # Fail silently if cache can't be written


def _dry_run_contradiction(claim: str, text: str) -> bool:
    """
    Heuristic-based contradiction detection for DRY_RUN mode.

    Looks for explicit negation keywords and temporal contradictions.
    """
    claim_lower = claim.lower()
    text_lower = text.lower()

    # Keywords that suggest contradiction
    contradiction_keywords = [
        "never",
        "not",
        "no",
        "didn't",
        "doesn't",
        "wasn't",
        "weren't",
        "cannot",
        "couldn't",
        "wouldn't",
        "shouldn't",
        "hadn't",
        "impossible",
        "false",
        "incorrect",
        "wrong",
    ]

    # Strong temporal contradiction signals
    temporal_negation = ["never", "first time", "never before", "for the first time"]
    temporal_repetition = [
        "years",
        "often",
        "frequently",
        "regularly",
        "repeatedly",
        "many times",
        "multiple",
        "several",
        "always",
    ]

    # Check for temporal contradictions: claim suggests repetition, text suggests first/never
    claim_suggests_repetition = any(word in claim_lower for word in temporal_repetition)
    text_suggests_first_time = any(phrase in text_lower for phrase in temporal_negation)

    if claim_suggests_repetition and text_suggests_first_time:
        return True

    # Check if text contains strong negation
    has_negation = any(keyword in text_lower for keyword in contradiction_keywords)

    # Extract key concepts from claim (simple word-based approach)
    claim_words = set(word.lower() for word in claim.split() if len(word) > 3)
    text_words = set(word.lower() for word in text.split() if len(word) > 3)

    # If there's negation and some word overlap, likely a contradiction
    if has_negation and len(claim_words & text_words) > 0:
        return True

    # Strong negation words like "never" are contradictions even without word overlap
    # if they relate to the same domain (travel, experience, etc.)
    if any(strong_neg in text_lower for strong_neg in ["never", "impossible", "false"]):
        # Check for semantic overlap in action/experience words
        action_words = {
            "travel",
            "traveling",
            "go",
            "went",
            "visit",
            "visited",
            "see",
            "saw",
            "experience",
            "experienced",
            "leave",
            "left",
        }
        claim_actions = action_words & set(claim_lower.split())
        text_actions = action_words & set(text_lower.split())

        if claim_actions and text_actions:
            return True

    return False


def count_contradictions(claim: str, texts: List[str]) -> int:
    """
    Count how many evidence chunks explicitly contradict the claim.

    Args:
        claim: The claim to evaluate
        texts: List of cleaned evidence text strings

    Returns:
        Number of chunks that contradict the claim
    """
    cache = _load_cache()
    contradiction_count = 0

    for text in texts:
        # Create cache key
        cache_key = f"{claim}|||{text}"

        # Check cache
        if cache_key in cache:
            is_contradiction = cache[cache_key]
        else:
            # Determine contradiction
            if DRY_RUN:
                is_contradiction = _dry_run_contradiction(claim, text)
            else:
                # Call LLM
                prompt = get_contradiction_prompt(claim, text)
                response = call_llama(prompt).upper().strip()

                # Parse YES/NO response
                is_contradiction = "NO" in response or response.startswith("N")

                # Cache result
                cache[cache_key] = is_contradiction
                _save_cache(cache)

        if is_contradiction:
            contradiction_count += 1

    return contradiction_count
