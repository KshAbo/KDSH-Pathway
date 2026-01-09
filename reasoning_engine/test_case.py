"""
Test case for the reasoning engine.
"""

import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from reasoning_engine.engine import evaluate_claim

claim = "He spent years traveling abroad during childhood."

evidence_chunks = [
    {
        "chunk_id": 1,
        "text": "This was the first time he had ever left his village.",
        "meta": {"book_name": "MockNovel", "position": 100}
    },
    {
        "chunk_id": 2,
        "text": "He had never seen the sea before that day.",
        "meta": {"book_name": "MockNovel", "position": 150}
    }
]

result = evaluate_claim(claim, evidence_chunks)
print(result)
