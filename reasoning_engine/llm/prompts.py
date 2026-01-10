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
    return f"""Answer ONLY YES or NO.

CHARACTER FOCUS:
This claim is ONLY about {character}. Ignore all other characters.
Only consider events, actions, or statements directly about {character}.
If the excerpt mentions other characters or situations not involving {character}, disregard them.

Claim:
{claim}

Excerpt from the novel:
{excerpt}

Does the excerpt clearly contradict the claim about {character}?"""


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
    return f"""Answer ONLY YES or NO.

CHARACTER FOCUS:
This claim is ONLY about {character}. Ignore all other characters.
Only consider events, actions, or statements directly about {character}.
If the excerpt mentions other characters or situations not involving {character}, disregard them.

Claim:
{claim}

Evidence from the story:
{excerpt}

Question:
Assuming both are true for {character}, do they describe a logical impossibility
in the character's abilities, actions, or life history?
Ignore differences in cause or explanation unless coexistence is impossible.

Answer ONLY YES or NO.
"""


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
