# reasoner.py

import json
from typing import Any, Dict, List
from query_generator import QueryGenerator
from retriever import Retriever
from indexing.novel_indexer import NovelIndexer

# import the actual reasoning engine
from reasoning_engine.engine import evaluate_claim


class Reasoner:
    """
    The Reasoner orchestrates:
        claims -> retrieval -> reasoning_engine.evaluate_claim -> output JSONL

    No shortcuts. No transformation of result structure.
    Does not implement DRY_RUN logic (that belongs inside reasoning_engine).
    """

    def __init__(
        self,
        indexer: NovelIndexer,
        book_name: str,
        top_k: int = 3,
        out_path: str = "out/results.jsonl"
    ):
        self.indexer = indexer
        self.book_name = book_name
        self.top_k = top_k
        self.out_path = out_path

        self.retriever = Retriever(indexer)
        self.query_gen = QueryGenerator()

    def _infer_character(self, claim: str) -> str:
        """
        Very soft heuristic for extracting a character from a claim.
        reasoning_engine itself can ignore if not used.
        """
        import re
        tokens = re.findall(r"\b([A-Z][a-z]{2,})\b", claim)
        return tokens[0] if tokens else ""

    def _convert_chunks(self, chunks) -> List[Dict[str, Any]]:
        """
        Convert NovelIndexer Chunk objects to evidence format expected by reasoning_engine:
            [{"chunk_id":..., "text":..., "meta":{...}}, ...]
        """
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
        Main loop:
        - iterate claims
        - retrieve evidence chunks
        - evaluate via reasoning_engine.evaluate_claim
        - write JSONL output to self.out_path
        """
        import os
        os.makedirs(os.path.dirname(self.out_path), exist_ok=True)

        with open(self.out_path, "w", encoding="utf-8") as fout:
            for q in self.query_gen.iter_claim_queries():
                claim = q["text"]
                cid = q["id"]
                cidx = q["claim_idx"]

                # retrieve chunks
                chunks = self.retriever.retrieve_chunks(self.book_name, claim, top_k=self.top_k)
                evidence = self._convert_chunks(chunks)

                # optional character context
                character = self._infer_character(claim)

                # reasoning_engine call
                result = evaluate_claim(
                    claim=claim,
                    evidence_chunks=evidence,
                    character=character,
                    return_rationale=True
                )

                # decorate final result for tracking pipeline lineage
                final = {
                    "id": cid,
                    "claim_idx": cidx,
                    "book_name": self.book_name,
                    "claim": claim,
                    "evaluation": result
                }

                fout.write(json.dumps(final, ensure_ascii=False) + "\n")

        print(f"[Reasoner] Completed: {self.out_path}")
