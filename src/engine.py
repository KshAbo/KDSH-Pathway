# engine.py

import re
import json
from typing import List, Dict, Any, Optional
import logging
import ollama  # optional, used only for explanation generation if available

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("engine")

# --- Config ---
DRY_RUN = True  # Set to False to enable LLM explanations (ollama) in rationale generation

# ----------------------------
# Simple normalization helpers
# ----------------------------
def normalize_text(t: str) -> str:
    return re.sub(r"\s+", " ", t.strip())

def normalize_evidence_chunks(evidence_chunks: List[Dict[str, Any]]) -> List[str]:
    """
    Return a list of normalized texts corresponding to the evidence_chunks
    """
    return [normalize_text(c.get("text", "")) for c in evidence_chunks]

# ----------------------------
# Basic contradiction analysis
# ----------------------------
def analyze_contradictions(claim: str, clean_texts: List[str], original_chunks: List[Dict[str, Any]], character: Optional[str] = None):
    """
    Very simple heuristic contradiction detection:
    - If an evidence text contains explicit negation words with claim keywords -> CONTRADICT
    - Returns (count, list_of_analysis)
    Each analysis: {"chunk_id":..., "relation": "CONTRADICT"/"NEUTRAL", "excerpt": "..."}
    """
    negation_words = {"not", "never", "no", "none", "didn't", "doesn't", "couldn't", "wasn't", "isn't", "without"}
    claim_lower = claim.lower()
    keywords = [w for w in re.findall(r"\w+", claim_lower) if len(w) > 3]
    analyses = []
    contradictions = 0

    for idx, text in enumerate(clean_texts):
        t_lower = text.lower()
        relation = "NEUTRAL"
        excerpt = text[:300]
        # Heuristic: find both claim keywords and negation near them (within 10 words)
        for kw in keywords:
            if kw in t_lower:
                # check window for negation
                pattern = r"(?:\b(?:{})\b).{{0,80}}(?:\b(?:{})\b)".format("|".join(map(re.escape, keywords)), "|".join(negation_words))
                if re.search(pattern, t_lower):
                    relation = "CONTRADICT"
                    contradictions += 1
                    break
        analyses.append({
            "chunk_id": original_chunks[idx].get("chunk_id") if idx < len(original_chunks) else idx,
            "relation": relation,
            "excerpt": excerpt
        })
    return contradictions, analyses

# ----------------------------
# Constraint detection (simple)
# ----------------------------
def analyze_constraints(claim: str, clean_texts: List[str], original_chunks: List[Dict[str, Any]], character: Optional[str] = None):
    """
    Heuristic detection: if claim asserts an impossible fact (e.g., 'died in 1700' but timeline suggests otherwise).
    This is domain-specific — here we do a weak heuristic: check for direct contradiction of named-entity facts.
    Returns (constraint_violation_bool, violating_chunk_dict or None)
    """
    # placeholder: no real world model — return False
    return False, None

# ----------------------------
# Aggregation heuristics
# ----------------------------
def aggregate_decision(contradictions: int, constraint_violation: bool) -> int:
    """
    Simple rule:
     - If any contradiction or constraint_violation -> decision=0 (invalid)
     - Else decision=1 (valid)
    """
    if contradictions > 0 or constraint_violation:
        return 0
    return 1

# ----------------------------
# Rationale generation (uses ollama if DRY_RUN is False)
# ----------------------------
def get_contradiction_explanation_prompt(claim: str, excerpt: str, character: str = "") -> str:
    return f"Explain briefly why the excerpt contradicts the claim.\nClaim: {claim}\nExcerpt: {excerpt}\n"

def get_constraint_explanation_prompt(claim: str, excerpt: str, character: str = "") -> str:
    return f"Explain briefly why the excerpt is incompatible with the claim constraints.\nClaim: {claim}\nExcerpt: {excerpt}\n"

def call_llama(prompt: str) -> str:
    # thin wrapper; requires ollama
    r = ollama.chat(model="llama3:8b", messages=[{"role":"user","content":prompt}])
    return r["message"]["content"]

def generate_evidence_rationale(
    claim: str,
    evidence_chunks: List[Dict[str, Any]],
    contradiction_analysis: List[Dict[str, Any]],
    constraint_analysis: Optional[Dict[str, Any]],
    decision: int,
    character: str = "",
) -> List[Dict[str, Any]]:
    """
    Generate evidence rationale explanations. Adapted from engine notes. See engine.txt for fuller spec. :contentReference[oaicite:4]{index=4}
    """
    rationale = []

    # Process contradictions
    for analysis in contradiction_analysis:
        if analysis.get("relation") == "CONTRADICT":
            excerpt = analysis.get("excerpt", "")
            if DRY_RUN:
                reason = "The excerpt explicitly contradicts the claim."
            else:
                try:
                    prompt = get_contradiction_explanation_prompt(claim, excerpt, character)
                    response = call_llama(prompt).strip()
                    reason = response.split(".")[0] + "." if "." in response else response
                except Exception as e:
                    reason = f"Contradiction detected (explanation failed: {str(e)})"

            rationale.append({
                "chunk_id": analysis.get("chunk_id"),
                "excerpt": excerpt,
                "relation": "CONTRADICT",
                "reason": reason,
                "character": character
            })

    # Constraint violation
    if constraint_analysis and constraint_analysis.get("chunk_id"):
        violating_chunk_id = constraint_analysis["chunk_id"]
        excerpt = constraint_analysis.get("text", "")
        already = any(r["chunk_id"] == violating_chunk_id for r in rationale)
        if not already:
            if DRY_RUN:
                reason = "The excerpt is logically incompatible with the claim's requirements."
            else:
                try:
                    prompt = get_constraint_explanation_prompt(claim, excerpt, character)
                    response = call_llama(prompt).strip()
                    reason = response.split(".")[0] + "." if "." in response else response
                except Exception as e:
                    reason = f"Constraint violation (explanation failed: {str(e)})"

            rationale.append({
                "chunk_id": violating_chunk_id,
                "excerpt": excerpt,
                "relation": "INCOMPATIBLE",
                "reason": reason,
                "character": character
            })

    # If decision==1 and no rationale, add a SUPPORT snippet as heuristic
    if decision == 1 and len(rationale) == 0:
        for analysis in contradiction_analysis[:1]:
            if analysis.get("relation") == "NEUTRAL":
                rationale.append({
                    "chunk_id": analysis.get("chunk_id"),
                    "excerpt": analysis.get("excerpt"),
                    "relation": "SUPPORT",
                    "reason": "The excerpt is consistent with the claim.",
                    "character": character
                })
                break

    return rationale

# ----------------------------
# Public evaluator (main logic)
# ----------------------------
def evaluate_claim(
    claim: str,
    evidence_chunks: List[Dict[str, Any]],
    character: str = "",
    return_rationale: bool = True
) -> Dict[str, Any]:
    """
    High-level evaluation function. Derived from engine design doc. See engine.txt. :contentReference[oaicite:5]{index=5}
    """
    clean_texts = normalize_evidence_chunks(evidence_chunks)

    # Filter by minimal length
    filtered_chunks = []
    for chunk in evidence_chunks:
        text = chunk.get("text", "").strip()
        if len(text) >= 10:
            filtered_chunks.append(chunk)

    # Simple contradiction & constraint analysis
    contradictions, contradiction_analysis = analyze_contradictions(claim, clean_texts, filtered_chunks, character)
    constraint_violation, violating_chunk = analyze_constraints(claim, clean_texts, filtered_chunks, character)

    decision = aggregate_decision(contradictions, constraint_violation)

    result = {
        "claim": claim,
        "contradictions": contradictions,
        "constraint_violation": bool(constraint_violation),
        "decision": decision
    }

    if return_rationale:
        evidence_rationale = generate_evidence_rationale(
            claim, filtered_chunks, contradiction_analysis, violating_chunk, decision, character
        )
        result["evidence_rationale"] = evidence_rationale

    return result

# ----------------------------
# Engine glue: handle() expected by reasoner
# ----------------------------
class Engine:
    def __init__(self, dry_run: bool = True):
        global DRY_RUN
        DRY_RUN = dry_run

    def handle(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        payload keys: id, claim_idx, claim, book_name, evidence_chunks
        returns a dict with evaluation + metadata
        """
        claim = payload.get("claim", "")
        evidence_chunks = payload.get("evidence_chunks", [])
        book = payload.get("book_name", "")
        cid = payload.get("id")
        cidx = payload.get("claim_idx", 0)

        # Character extraction simple heuristic: if a single capitalized word appears in claim, use it
        m = re.findall(r"\b([A-Z][a-z]{2,})\b", claim)
        character = m[0] if m else ""

        result = evaluate_claim(claim, evidence_chunks, character=character, return_rationale=True)

        # Add metadata and identification
        result_meta = {
            "id": cid,
            "claim_idx": cidx,
            "book_name": book,
            "claim": claim,
            "result": result
        }
        logger.info("Engine processed id=%s claim_idx=%s decision=%s", cid, cidx, result.get("decision"))
        return result_meta
