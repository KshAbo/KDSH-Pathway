"""
Constraint violation detection logic.
"""

import json
import os
from typing import List, Optional, Dict, Any, Tuple

from ..config import DRY_RUN, CACHE_ENABLED
from ..llm.llama_client import call_llama
from ..llm.prompts import get_constraint_compatibility_prompt

# Cache file path
CACHE_DIR = os.path.join(os.path.dirname(__file__), "..", "cache")
CACHE_FILE = os.path.join(CACHE_DIR, "constraint_cache.json")


def _load_cache() -> dict:
    """Load constraint cache from disk."""
    if not CACHE_ENABLED:
        # Minimal debug log when cache is globally disabled
        print("[DEBUG] Cache disabled — skipping constraint cache load")
        return {}

    if not os.path.exists(CACHE_FILE):
        return {}

    try:
        with open(CACHE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return {}


def _save_cache(cache: dict):
    """Save constraint cache to disk."""
    if not CACHE_ENABLED:
        # When cache is disabled, do not write cache
        print("[DEBUG] Cache disabled — skipping constraint cache write")
        return

    os.makedirs(CACHE_DIR, exist_ok=True)
    try:
        with open(CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(cache, f, indent=2, ensure_ascii=False)
    except IOError:
        pass  # Fail silently if cache can't be written


def supports_claim_llm(claim: str, text: str) -> bool:
    """
    Uses a constrained LLM call to determine whether the excerpt
    supports or reinforces the claim.
    Cached per (claim, text) pair.
    """
    cache = _load_cache()
    # Use a different cache key prefix for support checks
    cache_key = f"SUPPORT|||{claim}|||{text}"

    # Check cache
    if cache_key in cache:
        return cache[cache_key]

    # Only run LLM check if not in DRY_RUN mode
    if DRY_RUN:
        # In DRY_RUN, default to False (no support detected via heuristics)
        # This allows constraint checks to proceed normally
        return False

    prompt = f"""Answer ONLY YES or NO.

Claim:
{claim}

Excerpt:
{text}

Does the excerpt support or reinforce the claim?
"""

    response = call_llama(prompt).upper().strip()
    if not response:
        raise RuntimeError("Empty LLM response in support check")

    is_support = response.startswith("YES")

    # Cache result
    cache[cache_key] = is_support
    _save_cache(cache)

    return is_support


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
            # SUPPORT GATE — must run before constraint logic
            # Supporting evidence can never violate constraints
            if supports_claim_llm(claim, text):
                continue  # Skip constraint checks for supporting evidence

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


def analyze_constraints(
    claim: str, texts: List[str], chunks: List[Dict[str, Any]], character: str
) -> Tuple[bool, Optional[Dict[str, Any]]]:
    """
    Analyze constraint violations and return violation status with violating chunk.

    Args:
        claim: The claim to evaluate
        texts: List of cleaned evidence text strings (order matches chunks)
        chunks: List of evidence chunk dictionaries with chunk_id and text
        character: The character the claim is about

    Returns:
        Tuple of:
        - has_violation: bool (True if constraint violation detected)
        - violating_chunk: Optional[dict] with chunk_id and text if violation, None otherwise
    """
    cache = _load_cache()
    has_violation = False
    violating_chunk = None

    # Determine constraint violation (reuse existing logic)
    if DRY_RUN:
        # Use the same logic as detect_constraint_violation
        has_violation = _dry_run_constraint_violation(claim, texts)
        # Find first violating chunk in DRY_RUN mode (matches _dry_run_constraint_violation logic)
        if has_violation:
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
            for i, text in enumerate(texts):
                if i >= len(chunks):
                    break
                chunk = chunks[i]
                text_lower = text.lower()

                # Count constraint violation signals (matches _dry_run_constraint_violation logic)
                violation_signals = sum(
                    1 for keyword in constraint_keywords if keyword in text_lower
                )

                # Check for strong single signals (matches _dry_run_constraint_violation logic)
                strong_signal = any(
                    strong_signal in text_lower
                    for strong_signal in [
                        "first time",
                        "never before",
                        "for the first time",
                    ]
                )

                if violation_signals >= 2 or strong_signal:
                    violating_chunk = {
                        "chunk_id": chunk.get("chunk_id", i + 1),
                        "text": chunk.get("text", text),
                    }
                    break
    else:
        # Check compatibility with each text chunk
        # If any chunk is incompatible, we have a violation
        for i, text in enumerate(texts):
            if i >= len(chunks):
                break

            chunk = chunks[i]

            # SUPPORT GATE — must run before constraint logic
            # Supporting evidence can never violate constraints
            if supports_claim_llm(claim, text):
                continue  # Skip constraint checks for supporting evidence

            # Apply state-change gate before LLM check
            if not is_potentially_state_changing(text):
                continue

            # Create cache key per (claim, text) pair
            cache_key = f"{claim}|||{text}"

            # Check cache (reuse existing logic)
            if cache_key in cache:
                is_incompatible = cache[cache_key]
            else:
                print("[DEBUG] Calling LLM for constraint compatibility check")

                prompt = get_constraint_compatibility_prompt(claim, text, character)
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
                violating_chunk = {
                    "chunk_id": chunk.get("chunk_id", i + 1),
                    "text": chunk.get("text", text),
                }
                break

    return has_violation, violating_chunk


def check_constraints(key, *args, **kwargs):
    """
    Check constraints for the given key and arguments.

    This function checks the cache first, and if the result is not cached
    or if CACHE_ENABLED is False, it computes the constraint result
    using the provided arguments.

    Args:
        key: The cache key
        *args: Positional arguments for constraint computation
        **kwargs: Keyword arguments for constraint computation

    Returns:
        The computed constraint result
    """
    cache = _load_cache()
    if CACHE_ENABLED and key in cache:
        return cache[key]
    if not CACHE_ENABLED:
        print("[DEBUG] Cache disabled — recomputing constraint result")
    result = _compute_constraints(*args, **kwargs)
    _save_cache({**cache, key: result} if CACHE_ENABLED else cache)
    return result
