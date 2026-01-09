"""
Positive test case for the reasoning engine.
All evidence aligns with the claim.
"""

import sys
import os
from reasoning_engine.config import DRY_RUN

print(f"DRY_RUN: {DRY_RUN}")

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from reasoning_engine.engine import evaluate_claim

# -----------------------
# CLAIM
# -----------------------
claim = "He was handicapped since childhood and could not walk without assistance ever."

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
print(result)
