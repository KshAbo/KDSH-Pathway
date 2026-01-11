import pandas as pd
import json

# Paths
TEST_CSV = "data/test.csv"
CLAIMS_JSONL = "intermediate/test_claims.jsonl"
OUTPUT_CSV = "data/test_with_claims.csv"

# 1. Load original test.csv
df = pd.read_csv(TEST_CSV)

# Safety check
assert "id" in df.columns, "test.csv must have an 'id' column"

# 2. Load claims from JSONL into a dict
claims_map = {}

with open(CLAIMS_JSONL, "r") as f:
    for line in f:
        obj = json.loads(line)
        claims_map[obj["id"]] = obj["claims"]

# 3. Add claims column
def get_claims(row_id):
    return json.dumps(claims_map.get(row_id, []))

df["claims"] = df["id"].apply(get_claims)

# 4. Save new CSV
df.to_csv(OUTPUT_CSV, index=False)

print(f"Saved merged file to {OUTPUT_CSV}")
