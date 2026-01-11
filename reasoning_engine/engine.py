"""
Main reasoning engine entry point.
"""

from typing import List, Dict, Any, Optional

from .config import DRY_RUN
from .logic.normalize import normalize_evidence_chunks
from .logic.contradiction import count_contradictions
from .logic.constraints import detect_constraint_violation
from .logic.aggregation import aggregate_decision
from .llm.qwen_client import call_qwen
from .llm.prompts import (
    get_contradiction_explanation_prompt,
    get_constraint_explanation_prompt,
)


def generate_evidence_rationale(
    claim: str,
    evidence_chunks: List[Dict[str, Any]],
    contradiction_analysis: List[Dict[str, Any]],
    constraint_analysis: Optional[Dict[str, Any]],
    decision: int,
    character: str,
) -> List[Dict[str, Any]]:
    """
    Generate evidence rationale explanations.

    Only generates explanations for CONTRADICT and INCOMPATIBLE chunks.
    Explanations are generated AFTER decision is known and never influence the decision.
    SUPPORT and NEUTRAL chunks are excluded from rationale.

    Args:
        claim: The claim that was evaluated
        evidence_chunks: Original evidence chunks
        contradiction_analysis: List of analysis dicts from analyze_contradictions
        constraint_analysis: Optional dict with violating chunk from analyze_constraints
        decision: Final decision (0 or 1)
        character: The character name the claim is about

    Returns:
        List of evidence rationale units with chunk_id, excerpt, relation, reason, and character
    """
    rationale = []

    # Process contradictions
    for analysis in contradiction_analysis:
        if analysis["relation"] == "CONTRADICT":
            excerpt = analysis["excerpt"]

            if DRY_RUN:
                # Heuristic explanation for DRY_RUN mode
                reason = "The excerpt explicitly contradicts the claim."
            else:
                # Generate LLM explanation
                try:
                    prompt = get_contradiction_explanation_prompt(
                        claim, excerpt, character
                    )
                    response = call_qwen(prompt).strip()

                    if not response or response.startswith("__LLM_ERROR__"):
                        reason = "The excerpt logically contradicts the claim."
                    else:
                        # Extract first sentence only
                        reason = (
                            response.split(".")[0] + "."
                            if "." in response
                            else response
                        )
                        # Ensure single sentence
                        if len(reason) > 500:  # Sanity check
                            reason = reason[:497] + "..."
                except Exception as e:
                    reason = f"Contradiction detected (explanation failed: {str(e)})"

            rationale.append(
                {
                    "chunk_id": analysis["chunk_id"],
                    "excerpt": excerpt,
                    "relation": "CONTRADICT",
                    "reason": reason,
                    "character": character,
                }
            )

    # Process constraint violation
    if constraint_analysis and constraint_analysis.get("chunk_id"):
        violating_chunk_id = constraint_analysis["chunk_id"]
        excerpt = constraint_analysis["text"]

        # Only add if not already added as contradiction
        already_added = any(r["chunk_id"] == violating_chunk_id for r in rationale)

        if not already_added:
            if DRY_RUN:
                # Heuristic explanation for DRY_RUN mode
                reason = "The excerpt is logically incompatible with the claim's world-state requirements."
            else:
                # Generate LLM explanation
                try:
                    prompt = get_constraint_explanation_prompt(
                        claim, excerpt, character
                    )
                    response = call_qwen(prompt).strip()

                    if not response or response.startswith("__LLM_ERROR__"):
                        reason = "The excerpt logically contradicts the claim's implicit constraints."
                    else:
                        # Extract first sentence only
                        reason = (
                            response.split(".")[0] + "."
                            if "." in response
                            else response
                        )
                        # Ensure single sentence
                        if len(reason) > 500:  # Sanity check
                            reason = reason[:497] + "..."
                except Exception as e:
                    reason = (
                        f"Constraint violation detected (explanation failed: {str(e)})"
                    )

            rationale.append(
                {
                    "chunk_id": violating_chunk_id,
                    "excerpt": excerpt,
                    "relation": "INCOMPATIBLE",
                    "reason": reason,
                    "character": character,
                }
            )

    # Note: SUPPORT and NEUTRAL chunks are NOT included in evidence rationale
    return rationale


def evaluate_claim(
    claim: str,
    evidence_chunks: List[Dict[str, Any]],
    character: str,
    return_rationale: bool = False,
) -> Dict[str, Any]:
    """
    Evaluate whether a claim is consistent with evidence chunks.

    Args:
        claim: The claim string to evaluate
        evidence_chunks: List of evidence chunk dictionaries with:
            - chunk_id: int
            - text: str
            - meta: dict with book_name and position
        character: The character name the claim is about (REQUIRED)
        return_rationale: If True, include evidence_rationale in output (default: False)

    Returns:
        Dictionary with:
            - claim: str
            - contradictions: int
            - constraint_violation: bool
            - decision: int (1 = valid, 0 = invalid)
            - evidence_rationale: List[Dict] (only if return_rationale=True)
    """
    # Normalize evidence chunks
    clean_texts = normalize_evidence_chunks(evidence_chunks)

    # Filter chunks that passed normalization (matches normalize_evidence_chunks exactly)
    filtered_chunks = []
    for chunk in evidence_chunks:
        text = chunk.get("text", "").strip()
        if len(text) >= 10:  # MIN_CHUNK_LENGTH from normalize.py
            filtered_chunks.append(chunk)

    # Verify alignment (filtered_chunks and clean_texts should have same length and order)
    if len(filtered_chunks) != len(clean_texts):
        # If mismatch, match by text content (safety fallback)
        text_to_chunks = {}
        for chunk in filtered_chunks:
            text_key = chunk.get("text", "").strip()
            if text_key:
                if text_key not in text_to_chunks:
                    text_to_chunks[text_key] = []
                text_to_chunks[text_key].append(chunk)

        reordered_chunks = []
        for text in clean_texts:
            if text in text_to_chunks and text_to_chunks[text]:
                reordered_chunks.append(text_to_chunks[text].pop(0))

        if len(reordered_chunks) == len(clean_texts):
            filtered_chunks = reordered_chunks

    if return_rationale:
        # Use analyze functions to get both counts and analysis
        from .logic.contradiction import analyze_contradictions
        from .logic.constraints import analyze_constraints

        contradictions, contradiction_analysis = analyze_contradictions(
            claim, clean_texts, filtered_chunks, character
        )
        constraint_violation, violating_chunk = analyze_constraints(
            claim, clean_texts, filtered_chunks, character
        )

        # Aggregate decision
        decision = aggregate_decision(contradictions, constraint_violation)

        # Generate rationale AFTER decision is known
        constraint_analysis_dict = violating_chunk if violating_chunk else None
        evidence_rationale = generate_evidence_rationale(
            claim,
            filtered_chunks,
            contradiction_analysis,
            constraint_analysis_dict,
            decision,
            character,
        )

        return {
            "claim": claim,
            "contradictions": contradictions,
            "constraint_violation": constraint_violation,
            "decision": decision,
            "evidence_rationale": evidence_rationale,
        }
    else:
        # Default behavior: use existing functions (exact same as before)
        # Count contradictions
        contradictions = count_contradictions(claim, clean_texts, character)

        # Detect constraint violations
        constraint_violation = detect_constraint_violation(
            claim, clean_texts, character
        )

        # Aggregate decision
        decision = aggregate_decision(contradictions, constraint_violation)

        return {
            "claim": claim,
            "contradictions": contradictions,
            "constraint_violation": constraint_violation,
            "decision": decision,
        }
