from llm import chat


def extract_claims(backstory: str):
    prompt = f"""
Break the following backstory into atomic, testable claims.
Return a JSON list of strings only.

Backstory:
{backstory}
"""
    return chat(prompt)
