"""
run_monitor.py
--------------
Single entry point for the AI Data Quality Monitor.
Orchestrates the full pipeline in sequence:

    1. Data profiling      (profiler/profiler.py)
    2. GE validation       (expectations/taxi_suite.py)
    3. AI explanations     (ai/explainer.py)
    4. Slack alerting      (alerts/slack_alert.py)  ← coming next

Usage:
    python run_monitor.py
"""

import pandas as pd
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from profiler.profiler       import DataProfiler
from expectations.taxi_suite import run_ge_pipeline
from ai.explainer            import run_explainer


def run():
    """
    Executes all monitoring steps in sequence.
    Each step saves its output to reports/ as JSON.
    """

    print("\n" + "="*55)
    print("  AI DATA QUALITY MONITOR")
    print("="*55)

    # ── Step 1: Profiling ───────────────────────────────────
    print("\n[1/3] Running data profiler...")

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
    print("\n[2/3] Running Great Expectations validation...")

    ge_summary = run_ge_pipeline()

    print(f"\n  GE passed    : {ge_summary['passed']} / {ge_summary['total']}")
    print(f"  Success rate : {ge_summary['success_pct']}%")

    # ── Step 3: Claude API explanations ────────────────────
    print("\n[3/3] Running Claude API explanations...")

    explanations = run_explainer()

    issues_found = len(explanations.get("explanations", []))
    print(f"\n  Issues explained : {issues_found}")

    # ── Step 4 coming next ──────────────────────────────────
    print("\n[4/4] Slack alerting  — pending")

    print("\n" + "="*55)
    print("  Monitor run complete")
    print("="*55 + "\n")

    return {
        "profile"     : profile,
        "ge_summary"  : ge_summary,
        "explanations": explanations
    }


if __name__ == "__main__":
    run()