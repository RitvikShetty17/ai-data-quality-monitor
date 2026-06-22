import pandas as pd
import numpy as np

# ── 1. LOAD DATA ──────────────────────────────────────────────────────────────
print("=" * 60)
print("LOADING DATASET")
print("=" * 60)

# Read parquet file — much faster than CSV for large datasets
df = pd.read_parquet("data/yellow_tripdata_2024-01.parquet")

print(f"✓ Loaded successfully")
print(f"  Rows    : {df.shape[0]:,}")
print(f"  Columns : {df.shape[1]}")

# ── 2. COLUMN OVERVIEW ────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("COLUMN NAMES + DATA TYPES")
print("=" * 60)
print(df.dtypes.to_string())

# ── 3. FIRST 5 ROWS ───────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("FIRST 5 ROWS")
print("=" * 60)
pd.set_option("display.max_columns", None)   # show all columns
pd.set_option("display.width", 200)          # wider display
print(df.head())

# ── 4. BASIC STATISTICS ───────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("BASIC STATISTICS (numeric columns)")
print("=" * 60)
print(df.describe().round(2).to_string())

# ── 5. NULL CHECK ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("NULL VALUES PER COLUMN")
print("=" * 60)

null_counts = df.isnull().sum()
null_pct = (df.isnull().sum() / len(df) * 100).round(2)
null_report = pd.DataFrame({
    "null_count": null_counts,
    "null_pct"  : null_pct
})
# Only show columns that actually have nulls
nulls_present = null_report[null_report["null_count"] > 0]
if nulls_present.empty:
    print("No nulls found.")
else:
    print(nulls_present.to_string())

# ── 6. DUPLICATE CHECK ────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("DUPLICATE ROWS")
print("=" * 60)
dupe_count = df.duplicated().sum()
print(f"Duplicate rows found: {dupe_count:,}")

# ── 7. OBVIOUS QUALITY ISSUES ─────────────────────────────────────────────────
print("\n" + "=" * 60)
print("OBVIOUS DATA QUALITY ISSUES")
print("=" * 60)

# Negative trip distances (physically impossible)
neg_distance = (df["trip_distance"] < 0).sum()
print(f"Negative trip_distance       : {neg_distance:,}")

# Zero or negative fares (suspicious)
zero_fare = (df["fare_amount"] <= 0).sum()
print(f"Zero or negative fare_amount : {zero_fare:,}")

# Extremely high fares (possible outliers)
high_fare = (df["fare_amount"] > 500).sum()
print(f"fare_amount > $500           : {high_fare:,}")

# Zero passengers
zero_pass = (df["passenger_count"] == 0).sum()
print(f"passenger_count = 0          : {zero_pass:,}")

# Trip distance = 0 but fare > 0 (suspicious)
weird_trips = ((df["trip_distance"] == 0) & (df["fare_amount"] > 0)).sum()
print(f"trip_distance=0 but fare>0   : {weird_trips:,}")

# Pickup after dropoff (impossible timestamps)
if "tpep_pickup_datetime" in df.columns and "tpep_dropoff_datetime" in df.columns:
    time_issues = (df["tpep_pickup_datetime"] > df["tpep_dropoff_datetime"]).sum()
    print(f"Pickup time AFTER dropoff    : {time_issues:,}")

# ── 8. VALUE COUNTS FOR CATEGORICAL COLUMNS ───────────────────────────────────
print("\n" + "=" * 60)
print("CATEGORICAL COLUMN VALUE COUNTS")
print("=" * 60)

cat_cols = ["VendorID", "payment_type", "RatecodeID"]
for col in cat_cols:
    if col in df.columns:
        print(f"\n{col}:")
        print(df[col].value_counts(dropna=False).to_string())

print("\n" + "=" * 60)
print("EDA COMPLETE")
print("=" * 60)