"""
Evidence normalization and cleaning.
"""

from typing import List


def normalize_evidence_chunks(evidence_chunks: List[dict]) -> List[str]:
    """
    Clean and normalize evidence chunks.
    
    Removes very short chunks and trims whitespace.
    
    Args:
        evidence_chunks: List of chunk dictionaries with 'text' field
        
    Returns:
        List of clean text strings
    """
    MIN_CHUNK_LENGTH = 10  # Minimum characters for a valid chunk
    
    clean_texts = []
    for chunk in evidence_chunks:
        text = chunk.get("text", "").strip()
        
        # Skip very short chunks
        if len(text) >= MIN_CHUNK_LENGTH:
            clean_texts.append(text)
    
    return clean_texts
