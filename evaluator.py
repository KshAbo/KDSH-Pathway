from llm import chat


def evaluate_claim(claim, evidence):
    prompt = f"""
Claim:
{claim}

Textual Evidence:
{evidence}

Classify as exactly one:
Supported
Contradicted
Constrained
Unconstrained

Explain briefly.
"""
    return chat(prompt)


def score(label):
    mapping = {
        "Supported": 1,
        "Constrained": 0,
        "Unconstrained": 0,
        "Contradicted": -2
    }
    return mapping.get(label, 0)
