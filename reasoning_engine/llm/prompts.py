"""
Prompt templates for LLM interactions.
"""


def get_contradiction_prompt(claim: str, excerpt: str) -> str:
    """
    Generate prompt for contradiction detection.

    Args:
        claim: The claim to evaluate
        excerpt: The excerpt from the novel

    Returns:
        Formatted prompt string
    """
    return f"""Answer ONLY YES or NO.

Claim:
{claim}

Excerpt from the novel:
{excerpt}

Does the excerpt clearly contradict the claim?"""


def get_constraint_compatibility_prompt(claim: str, excerpt: str) -> str:
    """
    Generate prompt for constraint compatibility check.

    Args:
        claim: The claim to evaluate
        excerpt: The excerpt from the novel

    Returns:
        Formatted prompt string
    """
    return f"""Answer ONLY YES or NO.

Claim:
{claim}

Evidence from the story:
{excerpt}

Question:
Assuming both are true, do they describe a logical impossibility
in the character's abilities, actions, or life history?
Ignore differences in cause or explanation unless coexistence is impossible.

Answer ONLY YES or NO.
"""


def get_contradiction_explanation_prompt(claim: str, excerpt: str) -> str:
    return f"""Explain in ONE sentence why the excerpt contradicts the claim.
Do NOT add new facts.

Claim:
{claim}

Excerpt:
{excerpt}
"""


def get_constraint_explanation_prompt(claim: str, excerpt: str) -> str:
    return f"""Explain in ONE sentence why the excerpt is logically incompatible
with the claim, even if the claim is not explicitly denied.

Claim:
{claim}

Excerpt:
{excerpt}
"""
