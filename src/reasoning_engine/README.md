# Reasoning Engine

A standalone reasoning engine for evaluating claims against evidence chunks retrieved from novels.

## Overview

This subsystem evaluates whether a single claim is consistent with a set of evidence chunks. It focuses on:

- **Explicit contradiction detection**: Counts chunks that explicitly contradict the claim
- **Implicit constraint violation detection**: Detects impossible scenarios (e.g., "first time" vs. "years of travel")
- **Deterministic aggregation**: Pure logic-based decision making (no LLM for final decision)

## Architecture

```
reasoning_engine/
├── llm/
│   ├── llama_client.py        # Ollama HTTP client
│   ├── prompts.py             # Prompt templates
├── logic/
│   ├── normalize.py           # Evidence cleaning
│   ├── contradiction.py       # Contradiction detection
│   ├── constraints.py         # Constraint violation logic
│   ├── aggregation.py         # Decision rules
├── cache/
│   ├── contradiction_cache.json
│   ├── constraint_cache.json
├── engine.py                  # Main evaluate_claim()
├── test_case.py               # Runnable test
├── config.py                  # DRY_RUN flag
└── README.md
```

## Input Format

### Claim
```python
claim: str
```

### Evidence Chunks
```python
evidence_chunks: List[dict]

# Each chunk:
{
  "chunk_id": int,
  "text": str,
  "meta": {
      "book_name": str,
      "position": int
  }
}
```

## Output Format

```python
{
  "claim": str,
  "contradictions": int,
  "constraint_violation": bool,
  "decision": int   # 1 = valid, 0 = invalid
}
```

## Decision Logic

1. **Constraint violation** → `decision = 0` (immediate)
2. **≥ 2 contradictions** → `decision = 0`
3. **Otherwise** → `decision = 1`

## DRY_RUN Mode

When `DRY_RUN = True` in `config.py`:
- No LLM calls are made
- Uses deterministic heuristics
- Entire pipeline still runs

When `DRY_RUN = False`:
- Uses LLaMA-3 8B via Ollama (http://localhost:11434)
- All LLM outputs are cached to disk

## Usage

```python
from engine import evaluate_claim

result = evaluate_claim(claim, evidence_chunks)
print(result)
```

## Running Tests

```bash
cd reasoning_engine
python test_case.py
```

## Requirements

- Python 3.7+
- `requests` library (for Ollama HTTP calls)
- Ollama running locally with LLaMA-3 8B model (when DRY_RUN=False)

## Installation

```bash
pip install requests
```

## Design Principles

- **One chunk → one LLM call**: No multi-chunk prompts
- **No explanations from LLM**: Binary YES/NO responses only
- **Deterministic aggregation**: All final decisions are rule-based
- **Full caching**: All LLM outputs cached to disk
- **Testable without novels**: Works with mock data
