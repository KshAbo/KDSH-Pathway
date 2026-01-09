import pandas as pd
import json
from tqdm import tqdm
from claim_extraction import llama3_extract_claims

# Paths
TRAIN_PATH = "data/train.csv"
OUTPUT_PATH = "intermediate/train_claims.jsonl"

# Load dataset
df = pd.read_csv(TRAIN_PATH)

# Safety check
assert "id" in df.columns
assert "content" in df.columns

# Open output file
with open(OUTPUT_PATH, "w") as f:
    for _, row in tqdm(df.iterrows(), total=len(df)):
        backstory = str(row["content"]).strip()

        if not backstory:
            claims = []
        else:
            claims = llama3_extract_claims(backstory)

        record = {
            "id": int(row["id"]),
            "claims": claims
        }

        f.write(json.dumps(record) + "\n")

print(f"Saved extracted claims to {OUTPUT_PATH}")
