# query_generator.py

import json

CLAIM_JSONL = "intermediate/train_claims.jsonl"


class QueryGenerator:
    def __init__(self, jsonl_path=CLAIM_JSONL):
        self.jsonl_path = jsonl_path

    def iter_claim_queries(self):
        """
        Yields:
        { id: 46, idx: 0, claim: "..."}
        """
        with open(self.jsonl_path, "r") as f:
            for line in f:
                obj = json.loads(line)
                claim_id = obj["id"]
                for idx, claim in enumerate(obj["claims"]):
                    yield {
                        "id": claim_id,
                        "claim_idx": idx,
                        "text": claim
                    }
