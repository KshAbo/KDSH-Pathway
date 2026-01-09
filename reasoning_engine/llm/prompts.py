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

Statement A:
{claim}

Statement B:
{excerpt}

Can both statements be true in the same story world?"""
