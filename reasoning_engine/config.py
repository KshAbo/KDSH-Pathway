"""
Configuration module for the reasoning engine.
"""

# When True, use deterministic heuristics instead of LLM calls
DRY_RUN = True

# Ollama configuration
OLLAMA_ENDPOINT = "http://localhost:11434"
OLLAMA_MODEL = "llama3"
