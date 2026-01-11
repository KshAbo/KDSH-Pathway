# preprocess.py

"""
Standalone preprocess script.
No external dependency on claim_extraction.py or run_claim_extraction.py.

Pipeline:
1. Ingest and index novels using NovelIndexer
2. Extract claims from train.csv via qwen2.5:14b-instruct locally
3. Save claims JSONL for downstream reasoner/query_generator
"""

import os
import json
import time
import logging
import argparse
import re

import pandas as pd
from tqdm import tqdm
import ollama

# Try flexible import for NovelIndexer
NovelIndexer = None
try:
    from indexing.novel_indexer import NovelIndexer as _NI
    NovelIndexer = _NI
except Exception:
    try:
        from indexing.novel_indexer import NovelIndexer as _NI2
        NovelIndexer = _NI2
    except Exception:
        NovelIndexer = None


# ============================
# Config
# ============================

BOOKS_DIR = "./data/Books/"
REQUIRED_FILES = [
    "In Search of the Castaways.txt",
    "The Count of Monte Cristo.txt"
]

# You can change this to "data/test.csv" if you are processing test data now
TRAIN_PATH = "data/train.csv"
CLAIM_JSONL = "intermediate/train_claims.jsonl"

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("preprocess")


# ============================
# Inline Qwen Claim Extractor
# ============================

def qwen_extract_claims(text: str):
    """
    Inline claim extraction using qwen2.5:14b-instruct via ollama.chat.
    Produces atomic SVO factual statements.
    """

    prompt = f"""
Rewrite the text into a list of ATOMIC factual claims.

Rules:
- One fact per line
- Do NOT copy the original sentence structure
- Use simple subject-verb-object form
- Don't change character's original name or any proper nouns
- Do NOT add explanations or headings
- Do NOT add numbering

Text:
{text}
"""

    response = ollama.chat(
        model="qwen2.5:14b-instruct",
        messages=[{"role": "user", "content": prompt}]
    )

    raw = response["message"]["content"]

    claims = []

    for line in raw.split("\n"):
        line = line.strip()

        # Skip empty
        if not line:
            continue

        # Skip headers
        if "here is" in line.lower():
            continue

        # Remove numbering
        line = re.sub(r"^\d+[\.\)]\s*", "", line)

        # Skip junk
        if len(line) < 10:
            continue

        claims.append(line)

    return claims


# ============================
# Novel ingestion
# ============================

def process_novels():
    if NovelIndexer is None:
        raise RuntimeError("NovelIndexer not found. Ensure it is importable.")

    indexer = NovelIndexer()

    for fn in REQUIRED_FILES:
        path = os.path.join(BOOKS_DIR, fn)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing novel: {path}")

        book_name = fn.replace(".txt", "")
        logger.info(f"Ingesting '{book_name}'...")
        indexer.ingest(book_name, path)

    logger.info("Novel ingestion complete.")
    return indexer


# ============================
# Claim extraction
# ============================

def run_claim_extraction(retry=1):
    if not os.path.exists(TRAIN_PATH):
        raise FileNotFoundError(f"Dataset not found: {TRAIN_PATH}")

    df = pd.read_csv(TRAIN_PATH)

    if "id" not in df.columns or "content" not in df.columns:
        raise ValueError("CSV must contain columns: id, content")

    os.makedirs(os.path.dirname(CLAIM_JSONL), exist_ok=True)

    with open(CLAIM_JSONL, "w", encoding="utf-8") as fout:
        for _, row in tqdm(df.iterrows(), total=len(df)):
            backstory = str(row["content"]).strip() if not pd.isna(row["content"]) else ""

            attempts = 0
            claims = []

            while attempts <= retry:
                try:
                    # UPDATED: Using qwen_extract_claims now
                    claims = qwen_extract_claims(backstory) if backstory else []
                    break
                except Exception as e:
                    attempts += 1
                    logger.warning(f"Claim extraction failed for id={row['id']} | attempt {attempts}/{retry+1}: {e}")
                    time.sleep(1)
                    if attempts > retry:
                        logger.error(f"Giving up on id={row['id']} — using empty claims")
                        claims = []

            fout.write(json.dumps({"id": int(row["id"]), "claims": claims}, ensure_ascii=False) + "\n")

    logger.info(f"Claims saved → {CLAIM_JSONL}")
    return CLAIM_JSONL


# ============================
# CLI
# ============================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ingest-novels", action="store_true")
    parser.add_argument("--extract-claims", action="store_true")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--retry", default=1, type=int)
    args = parser.parse_args()

    if args.all or args.ingest_novels:
        process_novels()

    if args.all or args.extract_claims:
        run_claim_extraction(retry=args.retry)

    logger.info("Preprocess completed.")


if __name__ == "__main__":
    main()