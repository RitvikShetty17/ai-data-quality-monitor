"""
taxi_suite.py
-------------
Great Expectations validation suite for the NYC Taxi dataset.

Covers:
    - Null checks on critical columns
    - Value range checks on numeric columns
    - Categorical value checks
    - Dataset-level checks (row count, column count)
    - Column existence checks
    - Cross-column business logic checks

Run directly:
    python expectations/taxi_suite.py
"""

import great_expectations as gx
import pandas as pd
import json
import os
from datetime import datetime


def load_data() -> pd.DataFrame:
    """Load the NYC Taxi dataset."""
    print("📂 Loading dataset...")
    df = pd.read_parquet("data/yellow_tripdata_2024-01.parquet")
    print(f"   ✓ {len(df):,} rows loaded\n")
    return df


def build_suite(context) -> tuple:
    """
    Creates a fresh expectation suite.
    Deletes any existing suite with the same name first
    to avoid stale expectations.
    """
    suite_name = "nyc_taxi_quality_suite"

    try:
        context.delete_expectation_suite(suite_name)
    except Exception:
        pass

    suite = context.add_expectation_suite(
        expectation_suite_name=suite_name
    )

    return suite, suite_name


def add_all_expectations(validator) -> int:
    """
    Adds all expectations to the validator.

    Groups:
        1. Column existence     (3 expectations)
        2. Null checks          (6 expectations)
        3. Value range checks   (6 expectations)
        4. Categorical checks   (3 expectations)
        5. Dataset-level checks (2 expectations)
        6. Cross-column checks  (2 expectations)
                                ─────────────────
        Total                  (22 expectations)
    """
    count = 0

    # ── GROUP 1: COLUMN EXISTENCE ─────────────────────────────────────────────
    # Catches upstream schema changes — renamed or dropped columns
    critical_columns = [
        "VendorID",
        "tpep_pickup_datetime",
        "tpep_dropoff_datetime",
        "fare_amount",
        "total_amount",
        "trip_distance",
        "passenger_count",
        "payment_type",
        "PULocationID",
        "DOLocationID"
    ]

    validator.expect_table_columns_to_match_set(
        column_set=critical_columns,
        exact_match=False,   # allow extra columns, just ensure these exist
        meta={
            "group"      : "schema",
            "description": "All critical columns must be present in the dataset"
        }
    )
    count += 1

    # ── GROUP 2: NULL CHECKS ──────────────────────────────────────────────────
    # Core fields that must never be null for a record to be valid
    non_null_columns = [
        "VendorID",
        "tpep_pickup_datetime",
        "tpep_dropoff_datetime",
        "fare_amount",
        "total_amount",
        "trip_distance"
    ]

    for col in non_null_columns:
        validator.expect_column_values_to_not_be_null(
            column=col,
            meta={
                "group"      : "nulls",
                "description": f"{col} must never be null — required for every valid trip record"
            }
        )
        count += 1

    # ── GROUP 3: VALUE RANGE CHECKS ───────────────────────────────────────────
    range_checks = [
        {
            "column"     : "fare_amount",
            "min_value"  : 0.01,
            "max_value"  : 5000,
            "description": "Fare must be between $0.01 and $5000"
        },
        {
            "column"     : "total_amount",
            "min_value"  : 0.01,
            "max_value"  : 5000,
            "description": "Total amount must be between $0.01 and $5000"
        },
        {
            "column"     : "trip_distance",
            "min_value"  : 0,
            "max_value"  : 500,
            "description": "Trip distance must be between 0 and 500 miles"
        },
        {
            "column"     : "tip_amount",
            "min_value"  : 0,
            "max_value"  : 500,
            "description": "Tip amount must be between $0 and $500"
        },
        {
            "column"     : "tolls_amount",
            "min_value"  : 0,
            "max_value"  : 200,
            "description": "Tolls must be between $0 and $200"
        },
        {
            "column"     : "passenger_count",
            "min_value"  : 1,
            "max_value"  : 6,
            "mostly"     : 0.95,
            "description": "Passenger count must be 1-6 (NYC legal limit), allowing 5% for nulls"
        }
    ]

    for check in range_checks:
        kwargs = {
            "column"   : check["column"],
            "min_value": check["min_value"],
            "max_value": check["max_value"],
            "meta"     : {
                "group"      : "ranges",
                "description": check["description"]
            }
        }
        # Add mostly parameter if specified
        if "mostly" in check:
            kwargs["mostly"] = check["mostly"]

        validator.expect_column_values_to_be_between(**kwargs)
        count += 1

    # ── GROUP 4: CATEGORICAL VALUE CHECKS ─────────────────────────────────────
    # These columns should only ever contain known values
    # Any new value = something changed upstream

    # VendorID: 1 = Creative Mobile Technologies, 2 = VeriFone, 6 = unknown
    validator.expect_column_values_to_be_in_set(
        column="VendorID",
        value_set=[1, 2, 6],
        meta={
            "group"      : "categorical",
            "description": "VendorID must be 1 (CMT), 2 (VeriFone), or 6"
        }
    )
    count += 1

    # payment_type: 0=unknown, 1=credit, 2=cash, 3=no charge, 4=dispute
    validator.expect_column_values_to_be_in_set(
        column="payment_type",
        value_set=[0, 1, 2, 3, 4],
        meta={
            "group"      : "categorical",
            "description": "Payment type must be 0-4 per NYC TLC data dictionary"
        }
    )
    count += 1

    # store_and_fwd_flag: Y or N only (or null)
    validator.expect_column_values_to_be_in_set(
        column="store_and_fwd_flag",
        value_set=["Y", "N"],
        mostly=0.95,   # allow 5% for nulls we know exist
        meta={
            "group"      : "categorical",
            "description": "store_and_fwd_flag must be Y or N"
        }
    )
    count += 1

    # ── GROUP 5: DATASET-LEVEL CHECKS ─────────────────────────────────────────
    # Catch silent data loss — pipeline dropped rows or columns without warning

    # Row count must be above minimum (we know Jan 2024 has ~2.9M rows)
    validator.expect_table_row_count_to_be_between(
        min_value=1_000_000,   # if we ever get fewer than 1M rows something is wrong
        max_value=5_000_000,
        meta={
            "group"      : "dataset",
            "description": "Row count must be between 1M and 5M for a full month of NYC trips"
        }
    )
    count += 1

    # Column count must be exactly 19
    validator.expect_table_column_count_to_equal(
        value=19,
        meta={
            "group"      : "dataset",
            "description": "Dataset must have exactly 19 columns — catches schema drift"
        }
    )
    count += 1

    # ── GROUP 6: CROSS-COLUMN CHECKS ──────────────────────────────────────────
    # Business logic that involves two columns together
    # These are the most powerful checks — pure domain knowledge

    # PULocationID and DOLocationID must be valid NYC zone IDs (1-265)
    validator.expect_column_values_to_be_between(
        column="PULocationID",
        min_value=1,
        max_value=265,
        meta={
            "group"      : "cross_column",
            "description": "Pickup location must be a valid NYC taxi zone (1-265)"
        }
    )
    count += 1

    validator.expect_column_values_to_be_between(
        column="DOLocationID",
        min_value=1,
        max_value=265,
        meta={
            "group"      : "cross_column",
            "description": "Dropoff location must be a valid NYC taxi zone (1-265)"
        }
    )
    count += 1

    return count


def run_validation(context, df, suite_name) -> object:
    """
    Runs the full expectation suite against the dataset
    and returns the raw GE validation result object.
    """
    datasource    = context.get_datasource("nyc_taxi_datasource")
    asset         = datasource.get_asset("yellow_taxi_jan2024")
    batch_request = asset.build_batch_request(dataframe=df)

    validator = context.get_validator(
        batch_request=batch_request,
        expectation_suite_name=suite_name
    )

    # Add all expectations to the validator
    count = add_all_expectations(validator)

    # Save the suite before running
    validator.save_expectation_suite(discard_failed_expectations=False)
    print(f"  ✓ Suite saved with {count} expectations")

    # Run validation
    print("\n  Running validation against 2.9M rows...")
    validation_result = validator.validate()

    return validation_result, count


def parse_results(validation_result) -> dict:
    """
    Parses GE validation result into a clean structured dict
    suitable for saving as JSON and feeding to the Claude API.
    """
    stats       = validation_result.statistics
    passed      = int(stats["successful_expectations"])
    failed      = int(stats["unsuccessful_expectations"])
    total       = int(stats["evaluated_expectations"])
    success_pct = round(float(stats["success_percent"]), 1)

    print(f"\n{'='*55}")
    print(f"  GE VALIDATION RESULTS")
    print(f"{'='*55}")
    print(f"  Expectations run  : {total}")
    print(f"  Passed            : {passed}")
    print(f"  Failed            : {failed}")
    print(f"  Success rate      : {success_pct}%")
    print(f"\n  {'Expectation':<48} {'Status':<10} {'Violations'}")
    print(f"  {'-'*70}")

    parsed_results  = []
    failed_details  = []   # used by Claude API on Day 8

    for r in validation_result.results:
        exp_type = r.expectation_config.expectation_type
        kwargs   = r.expectation_config.kwargs
        col      = kwargs.get("column", "table-level")
        success  = r.success
        status   = "✅ PASS" if success else "❌ FAIL"
        meta     = r.expectation_config.meta or {}
        group    = meta.get("group", "general")

        # Extract violation details for failed checks
        violation_count = 0
        violation_pct   = 0.0

        if not success and hasattr(r, "result") and r.result:
            violation_count = r.result.get("unexpected_count", 0) or 0
            violation_pct   = round(
                float(r.result.get("unexpected_percent", 0) or 0), 2
            )

        label = f"{exp_type} [{col}]"
        viol  = f"{violation_count:,} ({violation_pct}%)" if not success else ""
        print(f"  {status}  {label:<48} {viol}")

        result_entry = {
            "expectation"    : exp_type,
            "column"         : col,
            "group"          : group,
            "passed"         : success,
            "violation_count": violation_count,
            "violation_pct"  : violation_pct,
            "description"    : meta.get("description", "")
        }

        parsed_results.append(result_entry)

        if not success:
            failed_details.append(result_entry)

    # Final summary
    ge_summary = {
        "validated_at"  : datetime.now().isoformat(),
        "total"         : total,
        "passed"        : passed,
        "failed"        : failed,
        "success_pct"   : success_pct,
        "results"       : parsed_results,
        "failed_details": failed_details   # Claude API reads this on Day 8
    }

    return ge_summary


def save_results(ge_summary: dict) -> str:
    """Saves GE results to reports/ as JSON."""
    os.makedirs("reports", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path      = f"reports/ge_results_{timestamp}.json"

    with open(path, "w") as f:
        json.dump(ge_summary, f, indent=2)

    print(f"\n  ✅ GE results saved → {path}")
    return path


def run_ge_pipeline() -> dict:
    """
    Public function called by run_monitor.py.
    Returns the full GE summary dict.
    """
    df      = load_data()
    context = gx.get_context(mode="file", project_root_dir=".")

    suite, suite_name         = build_suite(context)
    validation_result, count  = run_validation(context, df, suite_name)
    ge_summary                = parse_results(validation_result)
    save_results(ge_summary)

    return ge_summary


# ── ENTRY POINT ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    run_ge_pipeline()