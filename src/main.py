# src/main.py

"""
Unified entrypoint for the full pipeline.

This file ONLY orchestrates existing modules.
No logic is duplicated.
No teammate files are modified.
"""

import logging

# ----------------------------
# Import existing modules
# ----------------------------

# preprocess.py
import preprocess

# results.py (your reasoning runner)
import results

# ----------------------------
# Logging
# ----------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger("main")


def main():
    logger.info("========== PIPELINE START ==========")

    # ----------------------------------
    # Stage 1: Preprocessing
    # ----------------------------------
    logger.info("Running preprocess.py (--all)")
    
    # Reuse existing preprocess logic
    preprocess.process_novels()
    preprocess.run_claim_extraction()

    logger.info("Preprocessing completed.")

    # ----------------------------------
    # Stage 2: Reasoning
    # ----------------------------------
    logger.info("Running results.py")

    # results.py may execute at import-time OR expose main()
    if hasattr(results, "main"):
        results.main()
    else:
        logger.info("results.py executed via import")

    logger.info("========== PIPELINE COMPLETE ==========")


if __name__ == "__main__":
    main()
