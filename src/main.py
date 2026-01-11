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

# test.py (rename import-safe)
import test

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
    
    # Call preprocess main logic directly
    preprocess.process_novels()
    preprocess.run_claim_extraction()

    logger.info("Preprocessing completed.")

    # ----------------------------------
    # Stage 2: Reasoning
    # ----------------------------------
    logger.info("Running test.py")

    # test.py executes logic at import-time?
    # If not, we explicitly call its logic.

    if hasattr(test, "main"):
        test.main()
    else:
        # test.py runs at top-level (your current case)
        logger.info("test.py executed via import")

    logger.info("========== PIPELINE COMPLETE ==========")


if __name__ == "__main__":
    main()
