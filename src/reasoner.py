# reasoner.py

import json
from typing import Any
from query_generator import QueryGenerator
from retriever import Retriever
from indexing.novel_indexer import NovelIndexer

# Engine must expose a handle(payload) method (see engine.py below)
try:
    from engine import Engine
except Exception:
    Engine = None

class Reasoner:
    def __init__(self, indexer: NovelIndexer, engine_instance: Any = None, book_name: str = None, top_k: int = 3):
        self.indexer = indexer
        self.retriever = Retriever(indexer)
        self.query_gen = QueryGenerator()
        self.book_name = book_name
        self.top_k = top_k
        if engine_instance is None:
            if Engine is None:
                raise RuntimeError("Engine not available. Provide engine_instance or ensure engine.py is present.")
            engine_instance = Engine()
        self.engine = engine_instance

    def run(self, book_name: str = None, out_path: str = "out/results.jsonl"):
        """
        Run through all claim queries and evaluate by retrieving chunks and passing to engine.
        Saves engine outputs to out_path (one JSON per line).
        """
        if book_name is None and self.book_name is None:
            raise ValueError("book_name must be supplied (fixed book for retrieval)")
        book = book_name if book_name is not None else self.book_name

        import os
        os.makedirs(os.path.dirname(out_path), exist_ok=True)

        with open(out_path, "w", encoding="utf-8") as fout:
            for q in self.query_gen.iter_claim_queries():
                payload = {
                    "id": q["id"],
                    "claim_idx": q["claim_idx"],
                    "claim": q["text"],
                    "book_name": book
                }
                # Retrieve chunks (list of Chunk objects)
                chunks = self.retriever.retrieve_chunks(book, q["text"], top_k=self.top_k)
                # Convert chunks to dicts for engine
                evidence_chunks = []
                for c in chunks:
                    evidence_chunks.append({
                        "chunk_id": c.chunk_id,
                        "text": c.text,
                        "meta": {"book_name": c.book_name, "start_pos": c.start_pos, "end_pos": c.end_pos}
                    })

                payload["evidence_chunks"] = evidence_chunks

                # Pass to engine
                result = self.engine.handle(payload)
                # Write JSONL
                fout.write(json.dumps(result, ensure_ascii=False) + "\n")

        print(f"Reasoner run complete. Results saved to {out_path}")
