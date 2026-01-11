"""
Deterministic aggregation logic for final decision.
"""


def aggregate_decision(contradictions: int, constraint_violation: bool) -> int:
    """
    Aggregate contradictions and constraint violations into final decision.

    Rules:
    - If constraint_violation: decision = 0 (invalid)
    - Else if contradictions >= 2: decision = 0 (invalid)
    - Else: decision = 1 (valid)

    Args:
        contradictions: Number of chunks that contradict the claim
        constraint_violation: Whether constraint violation was detected

    Returns:
        1 if claim is valid, 0 if invalid
    """
    if constraint_violation:
        return 0
    elif contradictions >= 1:
        return 0
    else:
        return 1
