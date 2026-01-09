"""
Main reasoning engine entry point.
"""

from typing import List, Dict, Any

from .logic.normalize import normalize_evidence_chunks
from .logic.contradiction import count_contradictions
from .logic.constraints import detect_constraint_violation
from .logic.aggregation import aggregate_decision


def evaluate_claim(claim: str, evidence_chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Evaluate whether a claim is consistent with evidence chunks.
    
    Args:
        claim: The claim string to evaluate
        evidence_chunks: List of evidence chunk dictionaries with:
            - chunk_id: int
            - text: str
            - meta: dict with book_name and position
            
    Returns:
        Dictionary with:
            - claim: str
            - contradictions: int
            - constraint_violation: bool
            - decision: int (1 = valid, 0 = invalid)
    """
    # Normalize evidence chunks
    clean_texts = normalize_evidence_chunks(evidence_chunks)
    
    # Count contradictions
    contradictions = count_contradictions(claim, clean_texts)
    
    # Detect constraint violations
    constraint_violation = detect_constraint_violation(claim, clean_texts)
    
    # Aggregate decision
    decision = aggregate_decision(contradictions, constraint_violation)
    
    return {
        "claim": claim,
        "contradictions": contradictions,
        "constraint_violation": constraint_violation,
        "decision": decision
    }
