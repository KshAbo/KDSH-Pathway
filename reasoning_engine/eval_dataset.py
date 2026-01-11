import os
import time
import pandas as pd
from reasoning_engine.engine import evaluate_claim
from reasoning_engine.config import DRY_RUN, CACHE_ENABLED

DATA_PATH = os.path.join("data", "train_claims_with_evidence.csv")


def normalize_label(label: str) -> str:
    if pd.isna(label):
        return ""
    return str(label).strip().lower()


def run_dataset_evaluation():
    print(f"[CONFIG] DRY_RUN: {DRY_RUN}")
    print(f"[CONFIG] CACHE_ENABLED: {CACHE_ENABLED}")
    print()

    df = pd.read_csv(DATA_PATH)
    print(f"[INFO] Loaded {len(df)} rows from dataset")
    print(f"[INFO] Evaluating BACKSTORY-level accuracy\n")

    total_backstories = 0
    correct_backstories = 0
    total_claims_evaluated = 0

    current_book = None
    current_backstory_label = None
    predicted_contradict = False
    current_character = None

    start_time = time.time()

    for idx, row in df.iterrows():
        book_name = str(row.get("book_name", "")).strip()
        char = str(row.get("char", "")).strip()
        claim = str(row.get("claim", "")).strip()

        # Treat empty book_name as belonging to previous backstory
        if not book_name or book_name.lower() == "nan":
            book_name = ""

        # ---------- NEW BACKSTORY (only when book_name is non-empty AND different) ----------
        if book_name and book_name != current_book:
            # finalize previous backstory
            if current_book is not None:
                total_backstories += 1
                predicted_label = "contradict" if predicted_contradict else "consistent"
                match = "[OK]" if predicted_label == current_backstory_label else "[XX]"
                print(
                    f"[BACKSTORY {total_backstories:3d}] {current_book:35s} | Expected: {current_backstory_label:10s} | Predicted: {predicted_label:10s} {match}"
                )
                if predicted_label == current_backstory_label:
                    correct_backstories += 1

            # initialize new backstory
            current_book = book_name
            current_backstory_label = normalize_label(row.get("label", ""))
            predicted_contradict = False
            current_character = char if char else None
            print(
                f"\n>>> NEW BACKSTORY: {current_book} (Label: {current_backstory_label})"
            )

        # ---------- CHARACTER UPDATE ----------
        if char:
            current_character = char

        # ---------- END OF DATA ----------
        if not claim:
            break

        # ---------- SKIP IF ALREADY CONTRADICT ----------
        if predicted_contradict:
            continue

        # ---------- EXTRACT EVIDENCE ----------
        evidence_chunks = []
        for i in range(1, 6):
            ev = str(row.get(f"evidence_{i}", "")).strip()
            if ev and ev.lower() != "nan":
                evidence_chunks.append(
                    {
                        "chunk_id": i,
                        "text": ev,
                        "meta": {},
                    }
                )

        if not evidence_chunks:
            print(f"  [Claim skipped] No evidence chunks: {claim[:40]}...")
            continue

        total_claims_evaluated += 1

        # Progress every 10 claims
        if total_claims_evaluated % 10 == 1:
            elapsed = time.time() - start_time
            print(
                f"\n>>> PROGRESS: {total_claims_evaluated} claims evaluated ({elapsed:.1f}s)"
            )

        print(f"  [Claim {total_claims_evaluated}] Evaluating: {claim[:60]}...")

        result = evaluate_claim(
            claim=claim,
            evidence_chunks=evidence_chunks,
            character=current_character,
        )

        if result["decision"] == 0:
            predicted_contradict = True
            claim_verdict = "CONTRADICT"
            print(
                f"    -> {claim_verdict} (contradictions: {result['contradictions']}, constraint: {result['constraint_violation']})"
            )
        else:
            claim_verdict = "CONSISTENT"
            print(f"    -> {claim_verdict}")

    # ---------- FINAL BACKSTORY ----------
    if current_book is not None:
        total_backstories += 1
        predicted_label = "contradict" if predicted_contradict else "consistent"
        match = "[OK]" if predicted_label == current_backstory_label else "[XX]"
        print(
            f"[BACKSTORY {total_backstories:3d}] {current_book:35s} | Expected: {current_backstory_label:10s} | Predicted: {predicted_label:10s} {match}"
        )
        if predicted_label == current_backstory_label:
            correct_backstories += 1

    accuracy = correct_backstories / total_backstories if total_backstories else 0
    total_time = time.time() - start_time

    print()
    print("=" * 100)
    print("DATASET EVALUATION RESULTS (BACKSTORY LEVEL)")
    print("=" * 100)
    print(f"Total Backstories      : {total_backstories}")
    print(f"Total Claims Evaluated : {total_claims_evaluated}")
    print(f"Correct Predictions    : {correct_backstories}")
    print(f"Accuracy               : {accuracy:.4f}")
    print(f"Total Time             : {total_time:.2f}s")
    if total_backstories > 0:
        print(f"Time per Backstory     : {total_time/total_backstories:.2f}s")


if __name__ == "__main__":
    run_dataset_evaluation()
