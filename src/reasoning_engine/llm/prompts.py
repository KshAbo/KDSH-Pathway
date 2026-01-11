"""
Prompt templates for LLM interactions.
"""


def get_contradiction_prompt(claim: str, excerpt: str, character: str) -> str:
    """
    Generate prompt for contradiction detection.

    Args:
        claim: The claim to evaluate
        excerpt: The excerpt from the novel
        character: The character name this claim is about

    Returns:
        Formatted prompt string
    """
    return f"""Respond YES or NO only.

IMPORTANT: This evaluation is ONLY about {character}.
If the excerpt is NOT about {character}, respond NO.
If the excerpt is about OTHER CHARACTERS but not {character}, respond NO.
Only if the excerpt contradicts something about {character} specifically, respond YES.

Claim:
{claim}

Excerpt from the novel:
{excerpt}

Does the excerpt contradict the claim? Answer YES or NO."""


def get_constraint_compatibility_prompt(
    claim: str, excerpt: str, character: str
) -> str:
    """
    Generate prompt for constraint compatibility check.

    Args:
        claim: The claim to evaluate
        excerpt: The excerpt from the novel
        character: The character name this claim is about

    Returns:
        Formatted prompt string
    """
    return f"""Respond YES or NO only.

IMPORTANT: This evaluation is ONLY about {character}.
If the excerpt is NOT about {character}, respond NO.
If the excerpt is about OTHER CHARACTERS but not {character}, respond NO.
Only evaluate compatibility/incompatibility for {character}.

Claim:
{claim}

Evidence from the story:
{excerpt}

Question:
Assuming both are true for {character}, do they describe a logical impossibility
in the character's abilities, actions, or life history?

Answer YES or NO."""


def get_contradiction_explanation_prompt(
    claim: str, excerpt: str, character: str
) -> str:
    return f"""Explain in ONE sentence why the excerpt contradicts the claim about {character}.
Do NOT add new facts. Focus only on {character}.

Claim:
{claim}

Excerpt:
{excerpt}
"""


def get_constraint_explanation_prompt(claim: str, excerpt: str, character: str) -> str:
    return f"""Explain in ONE sentence why the excerpt is logically incompatible
with the claim about {character}, even if the claim is not explicitly denied.
Focus only on {character}.

Claim:
{claim}

Excerpt:
{excerpt}
"""
