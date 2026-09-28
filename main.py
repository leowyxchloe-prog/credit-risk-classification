import pandas as pd
import numpy as np

df = pd.read_csv("credit_risk.csv")

df_clean = df.copy()

# 1. Treat the invalid CLIENT_TYPE code as missing
df_clean["CLIENT_TYPE"] = df_clean["CLIENT_TYPE"].replace("0", pd.NA)

# 2. Explicitly preserve missing categorical information
categorical_missing_cols = [
    "INF_MARITAL_STATUS",
    "INF_GENDER",
    "CLIENT_TYPE",
    "COMPENSATION_CHARGED"
]

for col in categorical_missing_cols:
    df_clean[col] = df_clean[col].fillna("UNKNOWN")

df_clean["INSTALL_SIZE"] = df_clean["INSTALL_SIZE"].fillna(0)


print("Missing values after cleaning:")
print(df_clean.isna().sum())

print("\nCLIENT_TYPE:")
print(df_clean["CLIENT_TYPE"].value_counts(dropna=False))

print("Duplicate rows:", df_clean.duplicated().sum())
print("Duplicate account numbers:", df_clean["ACC_NO"].duplicated().sum())

print(df_clean["QUALITY_OF_LOAN"].value_counts())
print(
    df_clean["QUALITY_OF_LOAN"]
    .value_counts(normalize=True)
    .mul(100)
)

df_clean.info()
for col in df_clean.columns:
    print(f"\n--- {col} ---")
    print(df_clean[col].unique()[:20])

df_clean.to_csv("credit_risk_cleaned.csv", index=False)

numerical_cols = [
    "INVESTMENT_TOTAL",
    "ACCCURRENTBALANCE",
    "INSTALL_SIZE",
    "DUE_PAYMENT"
]

categorical_cols = [
    "INF_MARITAL_STATUS",
    "INF_GENDER",
    "COMPENSATION_CHARGED",
    "CLIENT_TYPE",
    "REPAY_MODE"
]

for col in categorical_cols:
    print(f"\n===== {col} =====")
    
    result = pd.crosstab(
        df_clean[col],
        df_clean["QUALITY_OF_LOAN"],
        normalize="index"
    ) * 100
    
    print(result)

id="2d6v8k"
for col in categorical_cols:
    print(f"\n===== {col} =====")
    
    result = pd.crosstab(
        df_clean[col],
        df_clean["QUALITY_OF_LOAN"]
    )
    
    result["Total"] = result.sum(axis=1)
    result["B_rate_%"] = (
        result["B"] / result["Total"] * 100
    )
    
    print(result)


for col in numerical_cols:
    zero_count = (df_clean[col] == 0).sum()
    zero_pct = zero_count / len(df_clean) * 100
    
    print(f"{col}: {zero_count} zeros ({zero_pct:.2f}%)")