"""
taxi_suite.py
-------------
Great Expectations suite for the NYC Taxi Jan 2024 dataset.

Contains 20+ expectations across:
- Null checks
- Value range checks  
- Business rule checks
- Cross-column checks
- Categorical value checks

Run this file to build + validate the suite:
    python expectations/taxi_suite.py
"""

import great_expectations as gx
import pandas as pd
import json
import os
from datetime import datetime


def load_data() -> pd.DataFrame:
    """Load the NYC Taxi dataset."""
    print("📂 Loading dataset for GE validation...")
    df = pd.read_parquet("data/yellow_tripdata_2024-01.parquet")
    print(f"   ✓ {len(df):,} rows loaded\n")
    return df


def build_suite(context) -> gx.core.ExpectationSuite:
    """
    Creates or updates the expectation suite.
    An expectation suite is simply a named collection of rules.
    """
    suite_name = "nyc_taxi_quality_suite"

    # Delete existing suite so we start fresh each time
    try:
        context.delete_expectation_suite(suite_name)
    except Exception:
        pass

    # Create a fresh suite
    suite = context.add_expectation_suite(
        expectation_suite_name=suite_name
    )

    print(f"  ✓ Suite created: {suite_name}")
    return suite, suite_name


def add_expectations(validator) -> int:
    """
    Adds all expectations to the validator.
    Each expect_*() call is one testable rule.
    Returns count of expectations added.
    """
    count = 0

    print("\n  Adding expectations...")

    # ── GROUP 1: NULL CHECKS (4 expectations) ────────────────────────────────
    print("\n  Group 1: Null checks")

    # Core fields should NEVER be null
    validator.expect_column_values_to_not_be_null(
        column="VendorID",
        meta={"group": "nulls", "description": "VendorID must always be present"}
    )
    count += 1

    validator.expect_column_values_to_not_be_null(
        column="tpep_pickup_datetime",
        meta={"group": "nulls", "description": "Pickup datetime must always be present"}
    )
    count += 1

    validator.expect_column_values_to_not_be_null(
        column="fare_amount",
        meta={"group": "nulls", "description": "Fare amount must always be present"}
    )
    count += 1

    validator.expect_column_values_to_not_be_null(
        column="total_amount",
        meta={"group": "nulls", "description": "Total amount must always be present"}
    )
    count += 1

    # ── GROUP 2: VALUE RANGE CHECKS (4 expectations) ─────────────────────────
    print("  Group 2: Value range checks")

    # Fare must be positive
    validator.expect_column_values_to_be_between(
        column="fare_amount",
        min_value=0.01,
        max_value=5000,
        meta={"group": "ranges", "description": "Fare amount must be between $0.01 and $5000"}
    )
    count += 1

    # Trip distance must be non-negative and realistic
    validator.expect_column_values_to_be_between(
        column="trip_distance",
        min_value=0,
        max_value=500,        # 500 miles is a generous upper limit for a taxi
        meta={"group": "ranges", "description": "Trip distance must be between 0 and 500 miles"}
    )
    count += 1

    # Passenger count must be within NYC legal limit
    validator.expect_column_values_to_be_between(
        column="passenger_count",
        min_value=1,
        max_value=6,
        mostly=0.95,          # allow 5% exceptions for nulls
        meta={"group": "ranges", "description": "Passenger count must be 1-6 (NYC legal limit)"}
    )
    count += 1

    # Total amount must be positive
    validator.expect_column_values_to_be_between(
        column="total_amount",
        min_value=0.01,
        max_value=5000,
        meta={"group": "ranges", "description": "Total amount must be positive"}
    )
    count += 1

    print(f"\n  ✓ {count} expectations added (Day 6 suite)")
    return count


def run_validation(context, df, suite_name):
    """
    Runs the expectation suite against the dataset
    and returns structured validation results.
    """
    print("\n" + "="*55)
    print("  Running GE validation...")
    print("="*55)

    # Get the datasource and asset we set up earlier
    datasource = context.get_datasource("nyc_taxi_datasource")
    asset      = datasource.get_asset("yellow_taxi_jan2024")

    # Build a batch request — tells GE which data to validate
    batch_request = asset.build_batch_request(dataframe=df)

    # Create a validator — this is the object that runs expectations
    validator = context.get_validator(
        batch_request=batch_request,
        expectation_suite_name=suite_name
    )

    return validator


def parse_and_print_results(validation_result) -> dict:
    """
    Parses GE's validation result into a clean summary
    and prints it to the terminal.
    """
    results   = validation_result.results
    stats     = validation_result.statistics

    passed    = int(stats["successful_expectations"])
    failed    = int(stats["unsuccessful_expectations"])
    total     = int(stats["evaluated_expectations"])
    success_pct = round(stats["success_percent"], 1)

    print(f"\n{'='*55}")
    print(f"  GE VALIDATION RESULTS")
    print(f"{'='*55}")
    print(f"  Expectations run    : {total}")
    print(f"  Passed              : {passed}")
    print(f"  Failed              : {failed}")
    print(f"  Success rate        : {success_pct}%")
    print(f"\n  {'Expectation':<45} {'Result'}")
    print(f"  {'-'*55}")

    ge_results = []

    for r in results:
        # Extract the expectation type and column
        exp_type  = r.expectation_config.expectation_type
        kwargs    = r.expectation_config.kwargs
        col       = kwargs.get("column", "multi-column")
        success   = r.success
        status    = "✅ PASS" if success else "❌ FAIL"

        # Get violation count if available
        if not success and hasattr(r, "result"):
            result_dict      = r.result
            unexpected_count = result_dict.get("unexpected_count", "N/A")
            unexpected_pct   = result_dict.get("unexpected_percent", 0)
            detail = f"{unexpected_count:,} violations ({round(unexpected_pct,2)}%)" if isinstance(unexpected_count, int) else ""
        else:
            detail = ""

        label = f"{exp_type} [{col}]"
        print(f"  {status}  {label:<43} {detail}")

        ge_results.append({
            "expectation"  : exp_type,
            "column"       : col,
            "passed"       : success,
            "detail"       : detail
        })

    # Build clean output dict
    return {
        "validated_at"   : datetime.now().isoformat(),
        "total"          : total,
        "passed"         : passed,
        "failed"         : failed,
        "success_pct"    : success_pct,
        "results"        : ge_results
    }


def save_ge_results(ge_summary: dict):
    """Saves GE results to reports/ as JSON."""
    os.makedirs("reports", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path      = f"reports/ge_results_{timestamp}.json"

    with open(path, "w") as f:
        json.dump(ge_summary, f, indent=2)

    print(f"\n  ✅ GE results saved → {path}")
    return path


# ── MAIN ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":

    # 1. Load data
    df = load_data()

    # 2. Get GE context
    context = gx.get_context(mode="file", project_root_dir=".")

    # 3. Build suite
    suite, suite_name = build_suite(context)

    # 4. Get validator + add expectations
    validator = run_validation(context, df, suite_name)
    count     = add_expectations(validator)

    # 5. Save suite
    validator.save_expectation_suite(discard_failed_expectations=False)
    print(f"\n  ✅ Suite saved with {count} expectations")

    # 6. Run validation
    validation_result = validator.validate()

    # 7. Parse + print results
    ge_summary = parse_and_print_results(validation_result)

    # 8. Save results JSON
    save_ge_results(ge_summary)

    print("\n" + "="*55)
    print("  DAY 6 complete — 8 expectations running")
    print("  DAY 7: expand to 20+ expectations")
    print("="*55 + "\n")