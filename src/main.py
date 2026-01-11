# src/main.py

"""
Unified entrypoint for the full pipeline.

Stages:
1. Ingest and index novels (Pathway-backed)
2. Extract claims from dataset using qwen2.5:14b-instruct
3. Run reasoning over extracted claims

Run with:
    python src/main.py
"""

import os
import logging

# ----------------------------
# Imports from your project
# ----------------------------

from preprocess import process_novels, run_claim_extraction
from indexing.novel_indexer import NovelIndexer
from reasoner import Reasoner

# ----------------------------
# Config
# ----------------------------

BOOKS_DIR = "./data/Books/"

NOVELS = [
    ("In search of the castaways", "In Search of the Castaways.txt"),
    ("The Count of Monte Cristo", "The Count of Monte Cristo.txt"),
]

OUTPUT_PATH = "out/results.jsonl"
TOP_K = 3

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("main")


# ----------------------------
# Main pipeline
# ----------------------------

def main():
    logger.info("========== PIPELINE START ==========")

    # --------------------------------------------------
    # Stage 1 + 2: Preprocessing (ingestion + claims)
    # --------------------------------------------------
    logger.info("Running preprocessing stage...")
    process_novels()
    run_claim_extraction()
    logger.info("Preprocessing completed.")

    # --------------------------------------------------
    # Stage 3: Reasoning
    # --------------------------------------------------
    logger.info("Initializing reasoning stage...")

    indexer = NovelIndexer()

    # NOTE: Index rebuild is expected for now
    for book_name, filename in NOVELS:
        path = os.path.join(BOOKS_DIR, filename)
        logger.info(f"Ingesting novel for reasoning: {book_name}")
        indexer.ingest(book_name, path)

    reasoner = Reasoner(
        indexer=indexer,
        book_name="The Count of Monte Cristo",
        top_k=TOP_K,
        out_path=OUTPUT_PATH
    )

    reasoner.run()

    logger.info("========== PIPELINE COMPLETE ==========")


if __name__ == "__main__":
    main()
