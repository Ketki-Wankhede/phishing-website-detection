
import pandas as pd
from pathlib import Path

file_path = Path("Phishing_Legitimate_full.csv")
df = pd.read_csv(file_path)

print("=" * 60)
print("1. DATASET OVERVIEW")
print("=" * 60)
print("Dataset shape:", df.shape)
print("\nFirst 5 rows:")
print(df.head().to_string())

print("\n" + "=" * 60)
print("2. COLUMN NAMES AND DATA TYPES")
print("=" * 60)
print(df.dtypes.to_string())

print("\n" + "=" * 60)
print("3. MISSING VALUES")
print("=" * 60)
print(df.isnull().sum().to_string())
print("Total missing values:", df.isnull().sum().sum())

print("\n" + "=" * 60)
print("4. DUPLICATE RECORDS")
print("=" * 60)
print("Exact duplicate rows:", df.duplicated().sum())

print("\n" + "=" * 60)
print("5. TARGET COLUMN CANDIDATES")
print("=" * 60)
for col in df.columns:
    if any(word in col.lower() for word in
           ["class", "label", "target", "result", "status"]):
        print(f"\nColumn: {col}")
        print(df[col].value_counts(dropna=False).to_string())

print("\n" + "=" * 60)
print("6. NUMERIC SUMMARY")
print("=" * 60)
print(df.describe().T.to_string())
