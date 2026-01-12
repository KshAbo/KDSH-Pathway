import json
import os
import re
import csv
from typing import Any, Dict, List
from tqdm import tqdm

# Internal imports
from retriever import Retriever
from indexing.novel_indexer import NovelIndexer
from reasoning_engine.engine import evaluate_claim

# ==========================================
# MACROS / CONFIGURATION
# ==========================================
CLAIM_PATH = "intermediate/train_claims.jsonl"
CLAIM_PATH_TESTING = "intermediate/test_claims.jsonl"


class Reasoner:
    """
    The Reasoner orchestrates:
        1. Reads claims from JSONL (grouped by ID)
        2. Iterates through partial claims for each ID
        3. Retrieves evidence & Evaluates
        4. Implements Short-Circuit AND Logic:
           - If ANY partial claim fails -> Extract reasons, save as FALSE, and Break.
           - If ALL partial claims pass -> Save as TRUE with standard reason.
        5. Saves output to both JSONL and CSV.
    """

    def __init__(
        self,
        indexer: NovelIndexer,
        book_name: str,
        claims_path: str = CLAIM_PATH,  # Defaults to Train, can pass CLAIM_PATH_TESTING
        top_k: int = 3,
        out_path: str = "out/results.jsonl"
    ):
        self.indexer = indexer
        self.book_name = book_name
        self.claims_path = claims_path
        self.top_k = top_k
        self.out_path = out_path

        self.retriever = Retriever(indexer)

    def _infer_character(self, claim: str) -> str:
        """Soft heuristic for extracting a character from a claim."""
        tokens = re.findall(r"\b([A-Z][a-z]{2,})\b", claim)
        return tokens[0] if tokens else ""

    def _convert_chunks(self, chunks) -> List[Dict[str, Any]]:
        """Convert NovelIndexer Chunk objects to evidence format."""
        evidence = []
        for c in chunks:
            evidence.append({
                "chunk_id": c.chunk_id,
                "text": c.text,
                "meta": {
                    "book_name": c.book_name,
                    "start_pos": c.start_pos,
                    "end_pos": c.end_pos
                }
            })
        return evidence

    def _write_csv_row(self, writer, obj):
        """Helper to write only story_id, prediction, and rationale to CSV."""
        row = {
            "story_id": obj["id"],
            "prediction": obj.get("decision", -1),
            "rationale": obj["Reason"]
        }
        writer.writerow(row)

    def run(self):
        """
        Main execution loop with Short-Circuit Logic and CSV Export.
        """
        # Ensure output directory exists
        os.makedirs(os.path.dirname(self.out_path), exist_ok=True)
        
        # Define CSV path based on out_path
        csv_path = self.out_path.replace(".jsonl", ".csv")

        # 1. Load all data first
        data_records = []
        if os.path.exists(self.claims_path):
            with open(self.claims_path, "r") as f:
                for line in f:
                    if line.strip():
                        data_records.append(json.loads(line))
        else:
            print(f"[Reasoner] Error: {self.claims_path} not found.")
            return

        print(f"[Reasoner] Processing {len(data_records)} IDs from {self.claims_path}...")
        print(f"[Reasoner] Saving to:\n  JSONL: {self.out_path}\n  CSV:   {csv_path}")

        # Open both files for writing
        with open(self.out_path, "w", encoding="utf-8") as f_json, \
             open(csv_path, "w", newline="", encoding="utf-8") as f_csv:
            
            # Setup CSV Writer with only the three required columns
            csv_headers = ["story_id", "prediction", "rationale"]
            csv_writer = csv.DictWriter(f_csv, fieldnames=csv_headers)
            csv_writer.writeheader()

            # Iterate over every ID
            for record in tqdm(data_records, desc="Verifying IDs"):
                row_id = record["id"]
                partial_claims = record["claims"]
                
                if not partial_claims:
                    continue

                all_passed = True

                # Nested Loop: Iterate through partial claims for this ID
                for idx, claim_text in enumerate(partial_claims):
                    
                    # 1. Retrieve
                    chunks = self.retriever.retrieve_chunks(self.book_name, claim_text, top_k=self.top_k)
                    evidence = self._convert_chunks(chunks)
                    character = self._infer_character(claim_text)

                    # 2. Evaluate
                    result = evaluate_claim(
                        claim=claim_text,
                        evidence_chunks=evidence,
                        character=character,
                        return_rationale=True
                    )

                    is_supported = (result.get("decision") == 1)

                    # 3. Check for Failure (AND Logic)
                    if not is_supported:
                        # === FAILURE CASE ===
                        
                        # Extract only the "reason" text from the list of dicts
                        raw_rationale_list = result.get("evidence_rationale", [])
                        reason_strings = [r.get("reason", "") for r in raw_rationale_list if "reason" in r]
                        
                        # Join them with ". "
                        combined_reason = ". ".join(reason_strings)

                        final_output_obj = {
                            "id": row_id,
                            "aggregate_status": "FALSE",
                            "decision": result.get("decision"),
                            "claim_idx": idx,
                            "total_claims": len(partial_claims),
                            "book_name": self.book_name,
                            "claim": claim_text,
                            "Reason": combined_reason  # Simplified reason string
                        }
                        
                        # Write to JSONL & CSV
                        f_json.write(json.dumps(final_output_obj, ensure_ascii=False) + "\n")
                        self._write_csv_row(csv_writer, final_output_obj)
                        
                        all_passed = False
                        break # <--- BREAK THE NESTED LOOP
                    
                # 4. Success Case (All Partial Claims Passed)
                if all_passed:
                    # === SUCCESS CASE ===
                    final_output_obj = {
                        "id": row_id,
                        "aggregate_status": "TRUE",
                        "decision": 1,
                        "claim_idx": len(partial_claims) - 1,
                        "total_claims": len(partial_claims),
                        "book_name": self.book_name,
                        "claim": "CONSISTENT",
                        "Reason": "All the claims in the backstory are true" # Standard success message
                    }
                    
                    # Write to JSONL & CSV
                    f_json.write(json.dumps(final_output_obj, ensure_ascii=False) + "\n")
                    self._write_csv_row(csv_writer, final_output_obj)

        print(f"[Reasoner] Completed.")