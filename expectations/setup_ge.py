"""
setup_ge.py
-----------
Creates the Great Expectations Data Context and configures
it to work with our NYC Taxi dataset.

Run once to set up GE:
    python expectations/setup_ge.py
"""

import great_expectations as gx
import pandas as pd
import os


def setup_data_context():
    """
    Creates a GE Data Context in the project root.
    This generates a gx/ folder with GE configuration.
    """
    print("\n" + "="*55)
    print("  Setting up Great Expectations Data Context")
    print("="*55)

    # Create a filesystem-based data context
    # This stores GE config in a local gx/ folder
    context = gx.get_context(mode="file", project_root_dir=".")

    print("  ✓ Data Context created → gx/ folder")
    print(f"  ✓ GE version: {gx.__version__}")

    return context


def add_dataframe_datasource(context):
    """
    Adds a Pandas DataFrame datasource to GE.
    This tells GE we'll be feeding it Pandas DataFrames.
    """
    print("\n  Adding Pandas datasource...")

    # Check if datasource already exists
    try:
        datasource = context.get_datasource("nyc_taxi_datasource")
        print("  ✓ Datasource already exists — skipping")
        return datasource
    except Exception:
        pass

    # Add a new pandas datasource
    datasource = context.sources.add_pandas(
        name="nyc_taxi_datasource"
    )

    print("  ✓ Pandas datasource added: nyc_taxi_datasource")
    return datasource


def add_data_asset(datasource):
    """
    Adds a data asset — a named reference to our dataset.
    Think of it as telling GE 'this is the table we want to test'.
    """
    print("\n  Adding data asset...")

    try:
        asset = datasource.get_asset("yellow_taxi_jan2024")
        print("  ✓ Data asset already exists — skipping")
        return asset
    except Exception:
        pass

    asset = datasource.add_dataframe_asset(
        name="yellow_taxi_jan2024"
    )

    print("  ✓ Data asset added: yellow_taxi_jan2024")
    return asset


if __name__ == "__main__":
    context   = setup_data_context()
    datasource = add_dataframe_datasource(context)
    asset      = add_data_asset(datasource)

    print("\n" + "="*55)
    print("  ✅ GE setup complete")
    print("  Next: run python expectations/taxi_suite.py")
    print("="*55 + "\n")