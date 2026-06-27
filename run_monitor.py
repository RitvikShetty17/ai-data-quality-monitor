"""
run_monitor.py
--------------
Single entry point for the AI Data Quality Monitor.
Orchestrates: profiling → validation → AI explanation → alerting

Usage:
    python run_monitor.py              # run once manually
"""

import pandas as pd
import sys
import os

# Make sure profiler module is importable
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from profiler.profiler import DataProfiler


def run():
    """
    Main pipeline — runs all monitoring steps in sequence.
    """
    print("\n" + "="*55)
    print("  AI DATA QUALITY MONITOR — starting run")
    print("="*55)

    # ── Step 1: Load data ───────────────────────────────────
    print("\n📂 Loading dataset...")
    df = pd.read_parquet("data/yellow_tripdata_2024-01.parquet")
    print(f"   ✓ {len(df):,} rows loaded")

    # ── Step 2: Run profiler ────────────────────────────────
    print("\n🔍 Running profiler...")
    cols_to_check = [
        "fare_amount", "trip_distance", "passenger_count",
        "tip_amount", "tolls_amount", "total_amount", "extra"
    ]
    profiler = DataProfiler(df, dataset_name="nyc_taxi_jan2024")
    report   = profiler.run_full_profile(numeric_cols=cols_to_check)

    # ── Steps 3-5 coming on Days 6-9 ───────────────────────
    print("\n⏳ Great Expectations validation  — coming Day 6")
    print("⏳ Claude API explanations        — coming Day 8")
    print("⏳ Slack alerting                 — coming Day 9")
    print("⏳ Streamlit dashboard            — coming Day 10")

    print("\n" + "="*55)
    print("  ✅ Monitor run complete")
    print("="*55 + "\n")

    return report


if __name__ == "__main__":
    run()