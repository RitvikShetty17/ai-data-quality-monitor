"""
profiler.py
-----------
Core data profiling engine for the AI Data Quality Monitor.
Handles: dataset overview, null detection, duplicate detection.
Output: structured Python dict (saved as JSON by run_monitor.py)
"""

import pandas as pd
import numpy as np
import json
import os
from datetime import datetime


class DataProfiler:
    """
    Profiles a Pandas DataFrame and returns structured quality metrics.
    
    Usage:
        profiler = DataProfiler(df, dataset_name="nyc_taxi")
        report   = profiler.run_full_profile()
    """

    def __init__(self, df: pd.DataFrame, dataset_name: str = "dataset"):
        """
        Parameters
        ----------
        df           : The DataFrame to profile
        dataset_name : A label used in the output report
        """
        self.df           = df
        self.dataset_name = dataset_name
        self.profile      = {}   # will hold the full report dict

    # ── 1. DATASET OVERVIEW ───────────────────────────────────────────────────
    def get_overview(self) -> dict:
        """
        Returns basic shape and memory info about the dataset.
        """
        overview = {
            "dataset_name"  : self.dataset_name,
            "profiled_at"   : datetime.now().isoformat(),
            "row_count"     : int(self.df.shape[0]),
            "column_count"  : int(self.df.shape[1]),
            "columns"       : list(self.df.columns),
            "dtypes"        : {
                col: str(dtype)
                for col, dtype in self.df.dtypes.items()
            },
            "memory_usage_mb": round(
                self.df.memory_usage(deep=True).sum() / 1024 / 1024, 2
            )
        }

        print(f"\n{'='*55}")
        print(f"  DATASET OVERVIEW — {self.dataset_name}")
        print(f"{'='*55}")
        print(f"  Rows         : {overview['row_count']:,}")
        print(f"  Columns      : {overview['column_count']}")
        print(f"  Memory       : {overview['memory_usage_mb']} MB")
        print(f"  Profiled at  : {overview['profiled_at']}")

        return overview

    # ── 2. NULL PROFILER ──────────────────────────────────────────────────────
    def get_null_profile(self) -> dict:
        """
        For every column, calculates null count and null percentage.
        Flags columns above a warning threshold (5%) and critical threshold (20%).
        """
        total_rows = len(self.df)
        null_data  = {}

        for col in self.df.columns:
            null_count = int(self.df[col].isnull().sum())
            null_pct   = round(null_count / total_rows * 100, 2)

            # Severity flag based on null percentage
            if null_pct == 0:
                severity = "ok"
            elif null_pct < 5:
                severity = "warning"
            elif null_pct < 20:
                severity = "critical"
            else:
                severity = "severe"

            null_data[col] = {
                "null_count" : null_count,
                "null_pct"   : null_pct,
                "severity"   : severity
            }

        # Summary counts
        cols_with_nulls    = sum(1 for v in null_data.values() if v["null_count"] > 0)
        cols_critical      = sum(1 for v in null_data.values() if v["severity"] in ("critical", "severe"))
        total_null_cells   = sum(v["null_count"] for v in null_data.values())

        print(f"\n{'='*55}")
        print(f"  NULL PROFILE")
        print(f"{'='*55}")
        print(f"  Columns with nulls   : {cols_with_nulls} / {len(self.df.columns)}")
        print(f"  Critical columns     : {cols_critical}")
        print(f"  Total null cells     : {total_null_cells:,}")
        print(f"\n  {'Column':<30} {'Null %':>8}  {'Severity'}")
        print(f"  {'-'*50}")
        for col, stats in null_data.items():
            if stats["null_count"] > 0:
                print(f"  {col:<30} {stats['null_pct']:>7}%  {stats['severity'].upper()}")

        return {
            "summary": {
                "columns_with_nulls" : cols_with_nulls,
                "columns_critical"   : cols_critical,
                "total_null_cells"   : total_null_cells
            },
            "by_column": null_data
        }

    # ── 3. DUPLICATE PROFILER ─────────────────────────────────────────────────
    def get_duplicate_profile(self) -> dict:
        """
        Detects fully duplicate rows and subset duplicates on key columns.
        """
        total_rows      = len(self.df)
        full_dupes      = int(self.df.duplicated().sum())
        full_dupe_pct   = round(full_dupes / total_rows * 100, 2)

        # Check for near-duplicates on key business columns
        # (same pickup time + same vendor = very likely duplicate booking)
        key_cols = [
            c for c in ["VendorID", "tpep_pickup_datetime", "trip_distance"]
            if c in self.df.columns
        ]
        if key_cols:
            key_dupes     = int(self.df.duplicated(subset=key_cols).sum())
            key_dupe_pct  = round(key_dupes / total_rows * 100, 2)
        else:
            key_dupes     = 0
            key_dupe_pct  = 0.0

        print(f"\n{'='*55}")
        print(f"  DUPLICATE PROFILE")
        print(f"{'='*55}")
        print(f"  Full duplicate rows   : {full_dupes:,}  ({full_dupe_pct}%)")
        print(f"  Key-column dupes      : {key_dupes:,}  ({key_dupe_pct}%)")
        print(f"  Key columns used      : {key_cols}")

        # New — checks both full and key-column dupes
        if full_dupes == 0 and key_dupes == 0:
            severity = "ok"
        elif full_dupes == 0 and key_dupe_pct < 2:
            severity = "warning"    # key dupes exist but under 2%
        elif full_dupes == 0 and key_dupe_pct >= 2:
            severity = "critical"   # significant key dupes
        else:
            severity = "critical"   # any full dupes = critical

        return {
            "full_duplicates"     : full_dupes,
            "full_duplicate_pct"  : full_dupe_pct,
            "key_column_dupes"    : key_dupes,
            "key_dupe_pct"        : key_dupe_pct,
            "key_columns_checked" : key_cols,
            "severity"            : severity
        }
# ── 3a. OUTLIER DETECTOR ─────────────────────────────────────────────────
    def get_outlier_profile(self, numeric_cols: list = None) -> dict:
        """
        Uses IQR (Interquartile Range) method to detect statistical outliers
        in numeric columns.

        IQR = Q3 - Q1
        Lower fence = Q1 - 1.5 * IQR  (anything below this = outlier)
        Upper fence = Q3 + 1.5 * IQR  (anything above this = outlier)

        Parameters
        ----------
        numeric_cols : list of column names to check.
                       If None, automatically picks all numeric columns.
        """

        # If no columns specified, pick all numeric ones automatically
        if numeric_cols is None:
            numeric_cols = self.df.select_dtypes(
                include=["int32", "int64", "float64"]
            ).columns.tolist()

        outlier_data = {}

        for col in numeric_cols:
            series = self.df[col].dropna()  # ignore nulls for this calculation

            # Calculate quartiles and IQR
            Q1  = series.quantile(0.25)
            Q3  = series.quantile(0.75)
            IQR = Q3 - Q1

            # Define the acceptable range
            lower_fence = Q1 - 1.5 * IQR
            upper_fence = Q3 + 1.5 * IQR

            # Count values outside the fences
            outliers_low  = int((series < lower_fence).sum())
            outliers_high = int((series > upper_fence).sum())
            total_outliers = outliers_low + outliers_high
            outlier_pct    = round(total_outliers / len(series) * 100, 2)

            # Assign severity
            if outlier_pct == 0:
                severity = "ok"
            elif outlier_pct < 1:
                severity = "warning"
            elif outlier_pct < 5:
                severity = "critical"
            else:
                severity = "severe"

            outlier_data[col] = {
                "q1"            : round(float(Q1), 4),
                "q3"            : round(float(Q3), 4),
                "iqr"           : round(float(IQR), 4),
                "lower_fence"   : round(float(lower_fence), 4),
                "upper_fence"   : round(float(upper_fence), 4),
                "outliers_low"  : outliers_low,
                "outliers_high" : outliers_high,
                "total_outliers": total_outliers,
                "outlier_pct"   : outlier_pct,
                "actual_min"    : round(float(series.min()), 4),
                "actual_max"    : round(float(series.max()), 4),
                "severity"      : severity
            }

        # Summary
        cols_with_outliers  = sum(1 for v in outlier_data.values() if v["total_outliers"] > 0)
        cols_severe         = sum(1 for v in outlier_data.values() if v["severity"] in ("critical","severe"))
        total_outlier_cells = sum(v["total_outliers"] for v in outlier_data.values())

        print(f"\n{'='*55}")
        print(f"  OUTLIER PROFILE (IQR method)")
        print(f"{'='*55}")
        print(f"  Columns checked        : {len(numeric_cols)}")
        print(f"  Columns with outliers  : {cols_with_outliers}")
        print(f"  Critical/severe cols   : {cols_severe}")
        print(f"\n  {'Column':<25} {'Outliers':>10}  {'Pct':>6}  {'Severity'}")
        print(f"  {'-'*55}")

        for col, stats in outlier_data.items():
            if stats["total_outliers"] > 0:
                print(
                    f"  {col:<25} {stats['total_outliers']:>10,}  "
                    f"{stats['outlier_pct']:>5}%  {stats['severity'].upper()}"
                )
                # Show the actual range vs allowed range for context
                print(
                    f"  {'':25}  actual: [{stats['actual_min']} → {stats['actual_max']}]"
                    f"  allowed: [{stats['lower_fence']} → {stats['upper_fence']}]"
                )

        return {
            "summary": {
                "columns_checked"       : len(numeric_cols),
                "columns_with_outliers" : cols_with_outliers,
                "columns_severe"        : cols_severe,
                "total_outlier_cells"   : total_outlier_cells
            },
            "by_column": outlier_data
        }

    # ── 3b. DATA TYPE / BUSINESS RULE CHECKER ────────────────────────────────
    def get_dtype_and_rule_profile(self) -> dict:
        """
        Checks dataset-specific business rules beyond just data types.
        For NYC Taxi data these are hard domain rules — things that should
        NEVER be true in valid data.

        These rules come directly from what we discovered in our Day 2 EDA.
        """

        rules = {}

        # ── Rule 1: Fare amount must be positive ─────────────────────────────
        if "fare_amount" in self.df.columns:
            violations = int((self.df["fare_amount"] <= 0).sum())
            rules["fare_amount_must_be_positive"] = {
                "column"      : "fare_amount",
                "rule"        : "fare_amount > 0",
                "violations"  : violations,
                "violation_pct": round(violations / len(self.df) * 100, 2),
                "severity"    : "ok" if violations == 0 else "critical"
            }

        # ── Rule 2: Trip distance must be non-negative ────────────────────────
        if "trip_distance" in self.df.columns:
            violations = int((self.df["trip_distance"] < 0).sum())
            rules["trip_distance_non_negative"] = {
                "column"      : "trip_distance",
                "rule"        : "trip_distance >= 0",
                "violations"  : violations,
                "violation_pct": round(violations / len(self.df) * 100, 2),
                "severity"    : "ok" if violations == 0 else "critical"
            }

        # ── Rule 3: Passenger count must be 1-6 (NYC legal limit) ────────────
        if "passenger_count" in self.df.columns:
            violations = int(
                ((self.df["passenger_count"] < 1) |
                 (self.df["passenger_count"] > 6)).sum()
            )
            rules["passenger_count_valid_range"] = {
                "column"      : "passenger_count",
                "rule"        : "1 <= passenger_count <= 6",
                "violations"  : violations,
                "violation_pct": round(violations / len(self.df) * 100, 2),
                "severity"    : "ok" if violations == 0 else "warning"
            }

        # ── Rule 4: Pickup must be before dropoff ─────────────────────────────
        if ("tpep_pickup_datetime" in self.df.columns and
                "tpep_dropoff_datetime" in self.df.columns):
            violations = int(
                (self.df["tpep_pickup_datetime"] >=
                 self.df["tpep_dropoff_datetime"]).sum()
            )
            rules["pickup_before_dropoff"] = {
                "column"      : "tpep_pickup_datetime + tpep_dropoff_datetime",
                "rule"        : "pickup_datetime < dropoff_datetime",
                "violations"  : violations,
                "violation_pct": round(violations / len(self.df) * 100, 2),
                "severity"    : "ok" if violations == 0 else "critical"
            }

        # ── Rule 5: Pickup year must be 2024 (dataset should only have 2024) ──
        if "tpep_pickup_datetime" in self.df.columns:
            violations = int(
                (self.df["tpep_pickup_datetime"].dt.year != 2024).sum()
            )
            rules["pickup_year_must_be_2024"] = {
                "column"      : "tpep_pickup_datetime",
                "rule"        : "year(pickup_datetime) == 2024",
                "violations"  : violations,
                "violation_pct": round(violations / len(self.df) * 100, 2),
                "severity"    : "ok" if violations == 0 else "critical"
            }

        # ── Rule 6: Zero distance but positive fare (suspicious) ──────────────
        if "trip_distance" in self.df.columns and "fare_amount" in self.df.columns:
            violations = int(
                ((self.df["trip_distance"] == 0) &
                 (self.df["fare_amount"] > 0)).sum()
            )
            rules["zero_distance_with_fare"] = {
                "column"      : "trip_distance + fare_amount",
                "rule"        : "NOT (trip_distance==0 AND fare_amount>0)",
                "violations"  : violations,
                "violation_pct": round(violations / len(self.df) * 100, 2),
                "severity"    : "ok" if violations == 0 else "warning"
            }

        # ── Rule 7: Total amount must equal components ─────────────────────────
        # NYC Taxi total = fare + extra + mta_tax + tip + tolls + 
        #                  improvement + congestion + airport_fee
        amount_cols = [
            "fare_amount", "extra", "mta_tax", "tip_amount",
            "tolls_amount", "improvement_surcharge",
            "congestion_surcharge", "Airport_fee"   # ← added these two
        ]
        if all(c in self.df.columns for c in amount_cols + ["total_amount"]):
            # Fill nulls with 0 for optional fee columns before summing
            calculated = self.df[amount_cols].fillna(0).sum(axis=1).round(2)
            actual     = self.df["total_amount"].round(2)
            # Allow $0.10 tolerance for floating point rounding
            violations = int((abs(calculated - actual) > 0.10).sum())
            rules["total_amount_matches_components"] = {
                "column"       : "total_amount",
                "rule"         : "total_amount ≈ sum of all fare components (±$0.10)",
                "violations"   : violations,
                "violation_pct": round(violations / len(self.df) * 100, 2),
                "severity"     : "ok" if violations == 0 else "warning"
            }

        # ── Print summary ─────────────────────────────────────────────────────
        total_violations = sum(r["violations"] for r in rules.values())
        rules_passed     = sum(1 for r in rules.values() if r["violations"] == 0)
        rules_failed     = len(rules) - rules_passed

        print(f"\n{'='*55}")
        print(f"  BUSINESS RULE CHECKS")
        print(f"{'='*55}")
        print(f"  Rules checked  : {len(rules)}")
        print(f"  Rules passed   : {rules_passed}")
        print(f"  Rules failed   : {rules_failed}")
        print(f"  Total violations: {total_violations:,}")
        print(f"\n  {'Rule':<40} {'Violations':>12}  {'Severity'}")
        print(f"  {'-'*60}")

        for rule_name, stats in rules.items():
            status = "✅" if stats["violations"] == 0 else "❌"
            print(
                f"  {status} {rule_name:<38} "
                f"{stats['violations']:>10,}  {stats['severity'].upper()}"
            )

        return {
            "summary": {
                "rules_checked"    : len(rules),
                "rules_passed"     : rules_passed,
                "rules_failed"     : rules_failed,
                "total_violations" : total_violations
            },
            "by_rule": rules
        }
    # ── 4. RUN FULL PROFILE ───────────────────────────────────────────────────
    # ── 4. RUN FULL PROFILE ───────────────────────────────────────────────────
    def run_full_profile(self, numeric_cols: list = None) -> dict:
        """
        Runs ALL profilers in sequence and saves combined JSON report.
        This is the single method you call to get the complete picture.
        """
        print("\n🔍 Starting full data profile...\n")

        self.profile = {
            "overview"      : self.get_overview(),
            "nulls"         : self.get_null_profile(),
            "duplicates"    : self.get_duplicate_profile(),
            "outliers"      : self.get_outlier_profile(numeric_cols),
            "business_rules": self.get_dtype_and_rule_profile()
        }

        # Calculate an overall health score (0-100)
        # Based on: rules passed, null severity, outlier severity
        rules        = self.profile["business_rules"]
        total_rules  = rules["summary"]["rules_checked"]
        passed_rules = rules["summary"]["rules_passed"]
        rule_score   = (passed_rules / total_rules * 100) if total_rules > 0 else 100

        null_cols_ok = sum(
            1 for v in self.profile["nulls"]["by_column"].values()
            if v["severity"] == "ok"
        )
        null_score = null_cols_ok / self.profile["overview"]["column_count"] * 100

        # Weighted score: 60% business rules, 40% null health
        health_score = round(0.6 * rule_score + 0.4 * null_score, 1)

        self.profile["health_score"] = {
            "score"      : health_score,
            "rule_score" : round(rule_score, 1),
            "null_score" : round(null_score, 1),
            "grade"      : (
                "A" if health_score >= 90 else
                "B" if health_score >= 75 else
                "C" if health_score >= 60 else
                "D"
            )
        }

        print(f"\n{'='*55}")
        print(f"  📊 OVERALL DATA HEALTH SCORE")
        print(f"{'='*55}")
        print(f"  Score        : {health_score} / 100")
        print(f"  Grade        : {self.profile['health_score']['grade']}")
        print(f"  Rule score   : {round(rule_score, 1)}%")
        print(f"  Null score   : {round(null_score, 1)}%")

        # Save JSON report
        os.makedirs("reports", exist_ok=True)
        timestamp   = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = f"reports/profile_{self.dataset_name}_{timestamp}.json"

        with open(report_path, "w") as f:
            json.dump(self.profile, f, indent=2, default=str)

        print(f"\n  ✅ Full profile saved → {report_path}")
        print(f"{'='*55}\n")

        return self.profile


# ── QUICK TEST ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Loading dataset...")
    df = pd.read_parquet("data/yellow_tripdata_2024-01.parquet")

    # We only profile the most meaningful numeric columns
    # (skipping location IDs since outlier detection on IDs is meaningless)
    cols_to_check = [
        "fare_amount", "trip_distance", "passenger_count",
        "tip_amount", "tolls_amount", "total_amount", "extra"
    ]

    profiler = DataProfiler(df, dataset_name="nyc_taxi_jan2024")
    report   = profiler.run_full_profile(numeric_cols=cols_to_check)
