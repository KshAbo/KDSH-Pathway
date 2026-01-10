"""
Diagnostic test case:
- 1 SUPPORT
- 1 CONTRADICTION
- 1 CONSTRAINT VIOLATION
- 1 NEUTRAL (other character)
"""

import sys
import os
import textwrap
from reasoning_engine.config import DRY_RUN

print(f"DRY_RUN: {DRY_RUN}")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from reasoning_engine.engine import evaluate_claim

# -----------------------
# CHARACTER
# -----------------------
character = "Elias"

# -----------------------
# CLAIM
# -----------------------
claim = "Elias had never left his hometown before adulthood."

# -----------------------
# EVIDENCE CHUNKS
# -----------------------
evidence_chunks = [
    {
        "chunk_id": 1,
        "text": (
            "Elias grew up in the same quiet riverside town where generations of his family "
            "had lived before him. As a child, he spent his days helping his mother at the "
            "market and listening to travelers describe distant lands he had never seen. "
            "He often wondered what lay beyond the hills but had no opportunity to leave."
        ),
        "meta": {"book_name": "MockNovel", "position": 400},
    },
    {
        "chunk_id": 2,
        "text": (
            "When Elias was twelve, he accompanied a merchant caravan across the southern "
            "pass, spending several weeks in the capital city before returning home. "
            "The journey left a deep impression on him and shaped his view of the wider world."
        ),
        "meta": {"book_name": "MockNovel", "position": 1200},
    },
    {
        "chunk_id": 3,
        "text": (
            "Elias spoke fluent coastal dialects with ease, recalling long childhood evenings "
            "spent among dockworkers in the port city, where he learned their customs and "
            "ways of life well before he came of age."
        ),
        "meta": {"book_name": "MockNovel", "position": 1800},
    },
    {
        "chunk_id": 4,
        "text": (
            "Marcus, Elias’s older brother, left their hometown at sixteen to join the navy. "
            "His travels took him across the sea, and he often returned with stories of "
            "foreign cities and distant wars."
        ),
        "meta": {"book_name": "MockNovel", "position": 2400},
    },
]

# -----------------------
# RUN EVALUATION
# -----------------------
result = evaluate_claim(claim, evidence_chunks, character)

print("\n" + "=" * 80)
print("EVALUATION RESULT")
print("=" * 80)
print(f"\nClaim:\n  {claim}")
print("\nDecision Metrics:")
print(f"  - Contradictions: {result['contradictions']}")
print(f"  - Constraint Violation: {result['constraint_violation']}")
print(
    f"  - Final Decision: {result['decision']} "
    f"({'INVALID' if result['decision'] == 0 else 'VALID'})"
)

if "evidence_rationale" in result:
    print("\n" + "=" * 80)
    print("EVIDENCE RATIONALE")
    print("=" * 80)
    for r in result["evidence_rationale"]:
        print(f"\nChunk {r['chunk_id']} — {r['relation']}")
        print(textwrap.fill(r["excerpt"], 76))
        print(f"Reason: {r['reason']}")
