# import ollama

# def llama3_extract_claims(text):
#     prompt = f"""
# Rewrite the text into a list of ATOMIC factual claims.

# Rules:
# - One fact per line
# - Do NOT copy the original sentence structure
# - Use simple subject-verb-object form
# - Don't change character's original name or any proper nouns

# Text:
# {text}
# """

#     response = ollama.chat(
#         model="llama3:8b",
#         messages=[{"role": "user", "content": prompt}]
#     )

#     raw = response["message"]["content"]

#     claims = [
#         line.strip("- ").strip()
#         for line in raw.split("\n")
#         if len(line.strip()) > 10
#     ]

#     return claims


# import re
# import ollama

# def llama3_extract_claims(text):
#     prompt = f"""
# Rewrite the text into a list of ATOMIC factual claims.

# Rules:
# - One fact per line
# - Do NOT copy the original sentence structure
# - Use simple subject-verb-object form
# - Don't change character's original name or any proper nouns
# - Do NOT add explanations or headings
# - Do NOT add numbering

# Text:
# {text}
# """

#     response = ollama.chat(
#         model="qwen2.5:14b-instruct",
#         messages=[{"role": "user", "content": prompt}]
#     )

#     raw = response["message"]["content"]

#     claims = []

#     for line in raw.split("\n"):
#         line = line.strip()

#         # ❌ Skip empty lines
#         if not line:
#             continue

#         # ❌ Skip header-like lines
#         if "here is" in line.lower():
#             continue

#         # ❌ Remove numbering like "1.", "2)", etc.
#         line = re.sub(r"^\d+[\.\)]\s*", "", line)

#         # ❌ Skip very short junk
#         if len(line) < 10:
#             continue

#         claims.append(line)

#     return claims

import re
import ollama

def qwen_extract_claims(text):
    prompt = f"""
Rewrite the text into a list of ATOMIC factual claims.

Rules:
- One fact per line
- Do NOT copy the original sentence structure
- Use simple subject-verb-object form
- Don't change character's original name or any proper nouns
- Do NOT add explanations or headings
- Do NOT add numbering

Text:
{text}
"""

    response = ollama.chat(
        model="qwen2.5:14b-instruct",
        messages=[{"role": "user", "content": prompt}]
    )

    raw = response["message"]["content"]

    claims = []

    for line in raw.split("\n"):
        line = line.strip()

        # Skip empty
        if not line:
            continue

        # Skip headers
        if "here is" in line.lower():
            continue

        # Remove numbering
        line = re.sub(r"^\d+[\.\)]\s*", "", line)

        # Skip junk
        if len(line) < 10:
            continue

        claims.append(line)

    return claims

# text = """
# Learning that Villefort meant to denounce him to Louis XVIII, Noirtier pre-emptively handed the conspiracy dossier to a British spy—the very file the Count of Monte Cristo later acquired—thereby engineering his son’s “lawful” murder."""

# claims = qwen_extract_claims(text)

# cnt=0;
# for c in claims:
#     if cnt==0:
#         cnt+=1
#         continue
#     print("-", c)
