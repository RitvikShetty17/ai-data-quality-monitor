"""
run_monitor.py
--------------
Single entry point for the AI Data Quality Monitor.
Orchestrates the full pipeline in sequence:

    1. Data profiling      (profiler/profiler.py)
    2. GE validation       (expectations/taxi_suite.py)
    3. AI explanations     (ai/explainer.py)         ← coming next
    4. Slack alerting      (alerts/slack_alert.py)   ← coming next

Usage:
    python run_monitor.py
"""

import pandas as pd
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from profiler.profiler      import DataProfiler
from expectations.taxi_suite import run_ge_pipeline


def run():
    """
    Executes all monitoring steps in sequence.
    Each step saves its output to reports/ as JSON.
    """

    print("\n" + "="*55)
    print("  AI DATA QUALITY MONITOR")
    print("="*55)

    # ── Step 1: Profiling ───────────────────────────────────
    print("\n[1/2] Running data profiler...")

    df = pd.read_parquet("data/yellow_tripdata_2024-01.parquet")

    cols_to_check = [
        "fare_amount", "trip_distance", "passenger_count",
        "tip_amount", "tolls_amount", "total_amount", "extra"
    ]

    profiler = DataProfiler(df, dataset_name="nyc_taxi_jan2024")
    profile  = profiler.run_full_profile(numeric_cols=cols_to_check)

    print(f"\n  Health score : {profile['health_score']['score']} / 100")
    print(f"  Grade        : {profile['health_score']['grade']}")

    # ── Step 2: GE Validation ───────────────────────────────
    print("\n[2/2] Running Great Expectations validation...")

    ge_summary = run_ge_pipeline()

    print(f"\n  GE expectations passed : {ge_summary['passed']} / {ge_summary['total']}")
    print(f"  GE success rate        : {ge_summary['success_pct']}%")

    # ── Steps 3-4 coming soon ───────────────────────────────
    print("\n[3/4] Claude API explanations  — pending")
    print("[4/4] Slack alerting           — pending")

    print("\n" + "="*55)
    print("  Monitor run complete")
    print("="*55 + "\n")

    return {
        "profile"   : profile,
        "ge_summary": ge_summary
    }


if __name__ == "__main__":
    run()