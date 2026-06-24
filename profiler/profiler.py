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

        severity = "ok" if full_dupes == 0 else (
            "warning" if full_dupe_pct < 1 else "critical"
        )

        return {
            "full_duplicates"     : full_dupes,
            "full_duplicate_pct"  : full_dupe_pct,
            "key_column_dupes"    : key_dupes,
            "key_dupe_pct"        : key_dupe_pct,
            "key_columns_checked" : key_cols,
            "severity"            : severity
        }

    # ── 4. RUN FULL PROFILE ───────────────────────────────────────────────────
    def run_full_profile(self) -> dict:
        """
        Runs all profilers and returns a single combined report dict.
        Also saves it as a JSON file to reports/.
        """
        print("\n🔍 Starting data profile...\n")

        self.profile = {
            "overview"   : self.get_overview(),
            "nulls"      : self.get_null_profile(),
            "duplicates" : self.get_duplicate_profile()
        }

        # Save JSON report to reports/ folder
        os.makedirs("reports", exist_ok=True)
        timestamp   = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = f"reports/profile_{self.dataset_name}_{timestamp}.json"

        with open(report_path, "w") as f:
            json.dump(self.profile, f, indent=2, default=str)

        print(f"\n{'='*55}")
        print(f"  ✅ Profile complete — saved to {report_path}")
        print(f"{'='*55}\n")

        return self.profile


# ── QUICK TEST ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Loading dataset...")
    df = pd.read_parquet("data/yellow_tripdata_2024-01.parquet")

    profiler = DataProfiler(df, dataset_name="nyc_taxi_jan2024")
    report   = profiler.run_full_profile()