import pandas as pd

df = pd.read_csv("data/train_claims_with_evidence.csv")
print("Columns:", df.columns.tolist())
print("Shape:", df.shape)
print("\nFirst row:")
for col in df.columns:
    val = str(df.iloc[0][col])[:150]
    print(f"  {col}: {val}")
