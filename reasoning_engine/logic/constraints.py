"""
Constraint violation detection logic.
"""

import json
import os
from typing import List

from ..config import DRY_RUN
from ..llm.llama_client import call_llama
from ..llm.prompts import get_constraint_compatibility_prompt

# Cache file path
CACHE_DIR = os.path.join(os.path.dirname(__file__), "..", "cache")
CACHE_FILE = os.path.join(CACHE_DIR, "constraint_cache.json")


def _load_cache() -> dict:
    """Load constraint cache from disk."""
    if not os.path.exists(CACHE_FILE):
        return {}

    try:
        with open(CACHE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return {}


def _save_cache(cache: dict):
    """Save constraint cache to disk."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    try:
        with open(CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(cache, f, indent=2, ensure_ascii=False)
    except IOError:
        pass  # Fail silently if cache can't be written


def is_potentially_state_changing(text: str) -> bool:
    """
    Structural (not semantic) filter.
    Detects whether text implies a persistent state, capability,
    commitment, or long-term condition.
    """
    t = text.lower()
    return any(
        phrase in t
        for phrase in [
            "for years",
            "throughout",
            "always",
            "never",
            "remained",
            "became",
            "served",
            "assigned",
            "trained",
            "educated",
            "appointed",
            "bound",
            "devoted",
            "stationed",
        ]
    )


def _dry_run_constraint_violation(claim: str, texts: List[str]) -> bool:
    """
    Heuristic-based constraint violation detection for DRY_RUN mode.

    Detects implicit impossibility signals like "first time", "never", etc.
    """
    claim_lower = claim.lower()

    # Keywords that suggest constraint violations
    constraint_keywords = [
        "first time",
        "never",
        "had not",
        "has not",
        "have not",
        "never before",
        "for the first time",
        "unfamiliar",
        "unknown",
        "unseen",
        "unheard",
        "never seen",
        "never heard",
    ]

    # Check if claim suggests repeated/ongoing activity
    claim_suggests_repetition = any(
        word in claim_lower
        for word in [
            "years",
            "often",
            "frequently",
            "regularly",
            "repeatedly",
            "many times",
            "multiple",
            "several",
            "always",
            "often",
        ]
    )

    if not claim_suggests_repetition:
        return False

    # Check if any evidence text contains constraint violation signals
    for text in texts:
        text_lower = text.lower()

        # Count constraint violation signals
        violation_signals = sum(
            1 for keyword in constraint_keywords if keyword in text_lower
        )

        # If we have multiple signals or strong signals, likely a violation
        if violation_signals >= 2:
            return True

        # Strong single signals
        if any(
            strong_signal in text_lower
            for strong_signal in ["first time", "never before", "for the first time"]
        ):
            return True

    return False


def detect_constraint_violation(claim: str, texts: List[str]) -> bool:
    """
    Detect if evidence violates implicit constraints of the claim.

    Args:
        claim: The claim to evaluate
        texts: List of cleaned evidence text strings

    Returns:
        True if constraint violation detected, False otherwise
    """
    cache = _load_cache()
    has_violation = False

    # Determine constraint violation
    if DRY_RUN:
        has_violation = _dry_run_constraint_violation(claim, texts)
    else:
        # Check compatibility with each text chunk
        # If any chunk is incompatible, we have a violation
        for text in texts:
            # Apply state-change gate before LLM check
            if not is_potentially_state_changing(text):
                continue

            # Create cache key per (claim, text) pair
            cache_key = f"{claim}|||{text}"

            # Check cache
            if cache_key in cache:
                is_incompatible = cache[cache_key]
            else:
                print("[DEBUG] Calling LLM for constraint compatibility check")

                prompt = get_constraint_compatibility_prompt(claim, text)
                response = call_llama(prompt).upper().strip()

                if not response:
                    raise RuntimeError("Empty LLM response in constraint check")

                # If LLM says NO (cannot be true together), we have a violation
                is_incompatible = response.startswith("NO")

                # Cache result
                cache[cache_key] = is_incompatible
                _save_cache(cache)

            if is_incompatible:
                has_violation = True
                break

    return has_violation
