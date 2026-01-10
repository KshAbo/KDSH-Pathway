"""
Positive test case for the reasoning engine.
All evidence aligns with the claim.
"""

import sys
import os
import textwrap
from reasoning_engine.config import DRY_RUN

print(f"DRY_RUN: {DRY_RUN}")

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from reasoning_engine.engine import evaluate_claim

# -----------------------
# CLAIM
# -----------------------
claim = "He was handicapped since his very childhood and couldn't walk without assistance of others."

# -----------------------
# EVIDENCE CHUNKS
# -----------------------
evidence_chunks = [
    {
        "chunk_id": 1,
        "text": (
            "From an early age, the boy suffered a serious injury to his right leg after a fall "
            "from a stone embankment near his village. The injury never healed properly, "
            "leaving him with chronic pain and severely limited mobility. Throughout his "
            "childhood, he relied on wooden crutches crafted by his father and later on a "
            "small hand-built cart to move longer distances. Villagers often slowed their pace "
            "to walk beside him, aware of how exhausting even short journeys could be."
        ),
        "meta": {"book_name": "MockNovel", "position": 500},
    },
    {
        "chunk_id": 2,
        "text": (
            "As he grew older, friends frequently helped him travel between neighboring "
            "settlements by pulling his cart along the dirt paths. While others walked or rode "
            "animals, he remained seated, grateful for their assistance. Long-distance travel "
            "was rare and carefully planned, as uneven ground caused him discomfort and "
            "fatigue. He never attempted journeys alone, knowing his physical limitations."
        ),
        "meta": {"book_name": "MockNovel", "position": 1200},
    },
    {
        "chunk_id": 3,
        "text": (
            "During the autumn festival, he spent most of the day seated near the marketplace, "
            "watching the crowds pass by. Friends brought him food and news from across the "
            "square, and musicians occasionally gathered nearby so he could listen without "
            "having to move. Witnesses remembered him smiling warmly from his place, rarely "
            "changing position and always relying on others for help when relocating."
        ),
        "meta": {"book_name": "MockNovel", "position": 2400},
    },
    {
        "chunk_id": 4,
        "text": (
            "That evening, as lanterns were lit and songs echoed softly through the streets, he "
            "reflected quietly on the life he had learned to accept. Though hardship had "
            "shaped much of his youth, he found comfort in companionship and the kindness "
            "of those who stood beside him whenever he needed help."
        ),
        "meta": {"book_name": "MockNovel", "position": 2600},
    },
]

# -----------------------
# RUN EVALUATION
# -----------------------
result = evaluate_claim(claim, evidence_chunks)

# Format output for readability
print("\n" + "=" * 80)
print("EVALUATION RESULT")
print("=" * 80)
print(f"\nClaim:")
print(
    f"  {textwrap.fill(result['claim'], width=76, initial_indent='', subsequent_indent='  ')}"
)
print(f"\nDecision Metrics:")
print(f"  - Contradictions: {result['contradictions']}")
print(f"  - Constraint Violation: {result['constraint_violation']}")
print(
    f"  - Final Decision: {result['decision']} ({'[INVALID]' if result['decision'] == 0 else '[VALID]'})"
)

if "evidence_rationale" in result and result["evidence_rationale"]:
    print(f"\n{'=' * 80}")
    print("EVIDENCE RATIONALE")
    print(f"{'=' * 80}")

    for idx, rationale_item in enumerate(result["evidence_rationale"], 1):
        print(f"\n{'-' * 80}")
        print(f"Evidence #{idx} (Chunk ID: {rationale_item['chunk_id']})")
        print(f"Relation: {rationale_item['relation']}")
        print(f"\nExcerpt:")
        # Wrap excerpt text at 76 characters with proper indentation
        excerpt_wrapped = textwrap.fill(
            rationale_item["excerpt"],
            width=76,
            initial_indent="  ",
            subsequent_indent="  ",
        )
        print(excerpt_wrapped)

        print(f"\nReason:")
        # Wrap reason text at 76 characters with proper indentation
        reason_wrapped = textwrap.fill(
            rationale_item["reason"],
            width=76,
            initial_indent="  ",
            subsequent_indent="  ",
        )
        print(reason_wrapped)
else:
    print("\nNo evidence rationale available.")

print("\n" + "=" * 80 + "\n")
