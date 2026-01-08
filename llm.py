import openai
from config import CHAT_MODEL

openai.api_key = None


def chat(prompt: str) -> str:
    response = openai.chat.completions.create(
        model=CHAT_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    return response.choices[0].message.content.strip()
