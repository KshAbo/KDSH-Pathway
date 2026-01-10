"""
Prompt templates for LLM interactions.
"""


def get_contradiction_prompt(claim: str, excerpt: str, character: str) -> str:
    """
    Generate prompt for contradiction detection.

    A contradiction exists ONLY if the excerpt makes the claim
    logically impossible to be true for the given character.
    """
    return f"""Answer ONLY YES or NO.

Claim:
{claim}

Excerpt:
{excerpt}

Question:
Does the excerpt explicitly assert a fact about "{character}"
that cannot coexist with the claim being true?

Answer ONLY YES or NO.
"""


def get_constraint_compatibility_prompt(
    claim: str, excerpt: str, character: str
) -> str:
    """
    Generate prompt for constraint compatibility check.

    A constraint violation exists ONLY if both statements
    cannot be true at the same time for the given character.
    """
    return f"""Answer ONLY YES or NO.

Claim:
{claim}

Excerpt:
{excerpt}

Question:
If both statements are true at the same time, do they require "{character}"
to exist in two logically incompatible world-states?

Answer ONLY YES or NO.
"""


def get_contradiction_explanation_prompt(
    claim: str, excerpt: str, character: str
) -> str:
    """
    Explain why a contradiction exists.
    """
    return f"""Explain in ONE sentence why the excerpt makes the claim about "{character}"
logically impossible to be true.

Claim:
{claim}

Excerpt:
{excerpt}
"""


def get_constraint_explanation_prompt(claim: str, excerpt: str, character: str) -> str:
    """
    Explain why a constraint violation exists.
    """
    return f"""Explain in ONE sentence why the excerpt is logically incompatible
with the claim about "{character}", even if the claim is not explicitly denied.

Claim:
{claim}

Excerpt:
{excerpt}
"""
