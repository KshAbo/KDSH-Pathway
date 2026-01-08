import json
from claims import extract_claims
from evaluator import evaluate_claim
from scoring import score_label

def predict_example(row, novel_store, top_k):
    claims = json.loads(extract_claims(row["content"]))

    total_score = 0
    contradicted = 0

    for claim in claims:
        evidence = novel_store.query(claim, top_k)
        evaluation = evaluate_claim(claim, evidence)

        label = evaluation.split()[0]
        total_score += score_label(label)

        if label == "Contradicted":
            contradicted += 1

    return {
        "score": total_score,
        "contradictions": contradicted
    }
