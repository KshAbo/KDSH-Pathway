import json
import os
import re
from typing import Any, Dict, List
from tqdm import tqdm

# Internal imports
from retriever import Retriever
from indexing.novel_indexer import NovelIndexer
from reasoning_engine.engine import evaluate_claim

class Reasoner:
    """
    The Reasoner orchestrates:
        1. Reads claims from JSONL (grouped by ID)
        2. Iterates through partial claims for each ID
        3. Retrieves evidence & Evaluates
        4. Implements Short-Circuit AND Logic:
           - If ANY partial claim fails -> Break and save failure.
           - If ALL partial claims pass -> Save success.
    """

    def __init__(
        self,
        indexer: NovelIndexer,
        book_name: str,
        claims_path: str = "intermediate/train_claims.jsonl",
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

    def run(self):
        """
        Main execution loop with Short-Circuit Logic.
        """
        os.makedirs(os.path.dirname(self.out_path), exist_ok=True)

        # 1. Load all data first to get total count for tqdm
        data_records = []
        if os.path.exists(self.claims_path):
            with open(self.claims_path, "r") as f:
                for line in f:
                    if line.strip():
                        data_records.append(json.loads(line))
        else:
            print(f"[Reasoner] Error: {self.claims_path} not found.")
            return

        print(f"[Reasoner] Processing {len(data_records)} IDs...")

        with open(self.out_path, "w", encoding="utf-8") as fout:
            
            # Iterate over every ID
            for record in tqdm(data_records, desc="Verifying IDs"):
                row_id = record["id"]
                partial_claims = record["claims"]
                
                # Default state if no claims exist
                if not partial_claims:
                    continue

                final_output_obj = None
                all_passed = True

                # Nested Loop: Iterate through partial claims for this ID
                for idx, claim_text in enumerate(partial_claims):
                    
                    # 1. Retrieve
                    chunks = self.retriever.retrieve_chunks(self.book_name, claim_text, top_k=self.top_k)
                    evidence = self._convert_chunks(chunks)
                    character = self._infer_character(claim_text)

                    # 2. Evaluate
                    # Assuming evaluate_claim returns a dict with 'decision' 
                    # where 1 = Supported, 0 = Refuted/NotEnoughtInfo
                    result = evaluate_claim(
                        claim=claim_text,
                        evidence_chunks=evidence,
                        character=character,
                        return_rationale=True
                    )

                    # Prepare the output object
                    final_output_obj = {
                        "id": row_id,
                        "claim_idx": idx,
                        "total_claims": len(partial_claims),
                        "book_name": self.book_name,
                        "claim": claim_text,
                        "evaluation": result,
                        "aggregate_status": "PENDING"
                    }

                    # 3. Check for Failure (AND Logic)
                    # We check if decision is NOT Supported (assuming 1 is Supported)
                    # You can adjust this condition based on your engine's specific output codes
                    is_supported = (result.get("decision") == 1)

                    if not is_supported:
                        # FAILURE CASE:
                        # One part is false -> The whole ID is false.
                        # We save this specific failure and BREAK.
                        final_output_obj["aggregate_status"] = "FALSE"
                        fout.write(json.dumps(final_output_obj, ensure_ascii=False) + "\n")
                        all_passed = False
                        break # <--- BREAK THE NESTED LOOP
                    
                    # If supported, we continue to the next partial claim in the loop

                # 4. Success Case
                # If the loop finished and all_passed is still True, we save the LAST result
                # marking the whole ID as True.
                if all_passed and final_output_obj is not None:
                    final_output_obj["aggregate_status"] = "TRUE"
                    fout.write(json.dumps(final_output_obj, ensure_ascii=False) + "\n")

        print(f"[Reasoner] Completed. Results saved to {self.out_path}")