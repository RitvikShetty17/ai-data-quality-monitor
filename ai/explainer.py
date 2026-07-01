"""
explainer.py
------------
AI explanation layer for the Data Quality Monitor.

Reads profiler and GE results from reports/, sends structured
findings to OpenAI GPT-4o, and returns plain-English explanations
with root cause analysis and recommended actions.

Each failed check gets its own explanation. Output is saved
as a JSON file to reports/.
"""

import json
import os
from datetime import datetime
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()


def load_latest_report(prefix: str) -> dict:
    """
    Loads the most recent JSON report matching a filename prefix
    from the reports/ directory.
    """
    reports_dir = "reports"

    matching = [
        f for f in os.listdir(reports_dir)
        if f.startswith(prefix) and f.endswith(".json")
    ]

    if not matching:
        raise FileNotFoundError(
            f"No report found with prefix '{prefix}' in {reports_dir}/"
        )

    matching.sort()
    latest = matching[-1]
    path   = os.path.join(reports_dir, latest)

    with open(path, "r") as f:
        data = json.load(f)

    print(f"   ✓ Loaded: {path}")
    return data


def build_context_summary(profile: dict, ge_summary: dict) -> str:
    """
    Builds a concise context string from both reports.
    Sent to the AI model — not raw data, just structured findings.
    """

    overview     = profile.get("overview", {})
    health       = profile.get("health_score", {})
    null_summary = profile.get("nulls", {}).get("summary", {})
    dupe_summary = profile.get("duplicates", {})
    rules        = profile.get("business_rules", {})
    outlier_sum  = profile.get("outliers", {}).get("summary", {})

    failed_rules = {
        name: stats
        for name, stats in profile.get("business_rules", {}).get("by_rule", {}).items()
        if stats["violations"] > 0
    }

    severe_outliers = {
        col: stats
        for col, stats in profile.get("outliers", {}).get("by_column", {}).items()
        if stats["severity"] in ("critical", "severe")
    }

    ge_failures = ge_summary.get("failed_details", [])

    context = f"""
DATASET OVERVIEW
----------------
Name         : {overview.get('dataset_name', 'unknown')}
Rows         : {overview.get('row_count', 0):,}
Columns      : {overview.get('column_count', 0)}
Profiled at  : {overview.get('profiled_at', 'unknown')}
Health score : {health.get('score', 0)} / 100  (Grade {health.get('grade', 'N/A')})

NULL PROFILE
------------
Columns with nulls : {null_summary.get('columns_with_nulls', 0)}
Total null cells   : {null_summary.get('total_null_cells', 0):,}

DUPLICATE PROFILE
-----------------
Full duplicates    : {dupe_summary.get('full_duplicates', 0):,}
Key-column dupes   : {dupe_summary.get('key_column_dupes', 0):,}  ({dupe_summary.get('key_dupe_pct', 0)}%)

OUTLIER PROFILE (IQR method)
-----------------------------
Columns with severe outliers : {outlier_sum.get('columns_severe', 0)}
"""

    for col, stats in severe_outliers.items():
        context += (
            f"  {col}: {stats['total_outliers']:,} outliers ({stats['outlier_pct']}%)  "
            f"actual range [{stats['actual_min']} → {stats['actual_max']}]  "
            f"expected [{stats['lower_fence']} → {stats['upper_fence']}]\n"
        )

    context += f"""
BUSINESS RULE VIOLATIONS
-------------------------
Rules checked : {rules.get('summary', {}).get('rules_checked', 0)}
Rules passed  : {rules.get('summary', {}).get('rules_passed', 0)}
Rules failed  : {rules.get('summary', {}).get('rules_failed', 0)}
"""

    for name, stats in failed_rules.items():
        context += (
            f"  FAILED: {name}\n"
            f"    Rule       : {stats['rule']}\n"
            f"    Violations : {stats['violations']:,}  ({stats['violation_pct']}%)\n"
            f"    Severity   : {stats['severity'].upper()}\n"
        )

    context += f"""
GREAT EXPECTATIONS VALIDATION
------------------------------
Expectations run  : {ge_summary.get('total', 0)}
Passed            : {ge_summary.get('passed', 0)}
Failed            : {ge_summary.get('failed', 0)}
Success rate      : {ge_summary.get('success_pct', 0)}%
"""

    for failure in ge_failures:
        context += (
            f"  FAILED: {failure['expectation']} on [{failure['column']}]\n"
            f"    Violations : {failure['violation_count']:,}  ({failure['violation_pct']}%)\n"
            f"    Rule desc  : {failure['description']}\n"
        )

    return context.strip()


def build_prompt(context_summary: str) -> str:
    """
    Builds the full prompt sent to the AI model.
    """
    return f"""You are a senior data analyst reviewing an automated data quality report for a NYC Yellow Taxi dataset (January 2024, 2.96 million trip records).

Below are the findings from two independent data quality tools — a custom Python profiler and Great Expectations. Your job is to analyse these findings and produce clear, actionable explanations for the data team.

{context_summary}

---

For each distinct data quality issue found above, provide an explanation in this exact JSON structure:

{{
  "explanations": [
    {{
      "issue"             : "short name of the issue",
      "column"            : "affected column(s)",
      "severity"          : "critical | warning | info",
      "violation_count"   : number,
      "what_happened"     : "plain-English explanation of what the data shows and why it is a problem",
      "likely_cause"      : "the most probable root cause based on common data engineering patterns",
      "business_impact"   : "how this affects downstream analysis, reporting, or business decisions if left unfixed",
      "recommended_action": "specific, actionable step the data team should take to investigate or fix this"
    }}
  ],
  "overall_assessment": "2-3 sentence summary of the dataset overall health and the most urgent issue to address first"
}}

Return only valid JSON. No markdown, no code blocks, no preamble. Start your response with {{ and end with }}.
"""


def call_ai_api(prompt: str) -> dict:
    """
    Sends the prompt to OpenAI GPT-4o and parses the JSON response.
    """
    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY not found. "
            "Check your .env file has the key set correctly."
        )

    client = OpenAI(api_key=api_key)

    print("   Sending findings to GPT-4o...")

    response = client.chat.completions.create(
        model="gpt-4o",
        max_tokens=4096,
        temperature=0.2,       # low temperature = consistent, factual responses
        messages=[
            {
                "role"   : "system",
                "content": "You are a senior data analyst. Always respond with valid JSON only."
            },
            {
                "role"   : "user",
                "content": prompt
            }
        ]
    )

    response_text = response.choices[0].message.content.strip()

    print(f"   ✓ Response received ({len(response_text):,} characters)")
    print(f"   Tokens used : {response.usage.total_tokens:,}")

    # Strip markdown code fences if model wrapped response in them
    if response_text.startswith("```"):
        response_text = response_text.split("```")[1]
        if response_text.startswith("json"):
            response_text = response_text[4:]
        response_text = response_text.strip()

    try:
        explanations = json.loads(response_text)
    except json.JSONDecodeError as e:
        print(f"   ⚠ JSON parse error: {e}")
        explanations = {
            "raw_response": response_text,
            "parse_error" : str(e)
        }

    return explanations


def print_explanations(explanations: dict):
    """
    Prints AI explanations in a readable terminal format.
    """
    print(f"\n{'='*55}")
    print(f"  AI — DATA QUALITY EXPLANATIONS")
    print(f"{'='*55}")

    if "overall_assessment" in explanations:
        print(f"\n  📋 Overall Assessment")
        print(f"  {'-'*50}")
        print(f"  {explanations['overall_assessment']}\n")

    for i, exp in enumerate(explanations.get("explanations", []), 1):
        severity_icon = {
            "critical": "🔴",
            "warning" : "🟡",
            "info"    : "🔵"
        }.get(exp.get("severity", "info"), "⚪")

        print(f"  {severity_icon} Issue {i}: {exp.get('issue', 'Unknown')}")
        print(f"  {'─'*50}")
        print(f"  Column    : {exp.get('column', 'N/A')}")
        print(f"  Severity  : {exp.get('severity', 'N/A').upper()}")
        print(f"  Violations: {exp.get('violation_count', 0):,}")
        print(f"\n  What happened:")
        print(f"  {exp.get('what_happened', '')}")
        print(f"\n  Likely cause:")
        print(f"  {exp.get('likely_cause', '')}")
        print(f"\n  Business impact:")
        print(f"  {exp.get('business_impact', '')}")
        print(f"\n  Recommended action:")
        print(f"  {exp.get('recommended_action', '')}")
        print()


def save_explanations(explanations: dict) -> str:
    """Saves AI explanations to reports/ as JSON."""
    os.makedirs("reports", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path      = f"reports/ai_explanations_{timestamp}.json"

    with open(path, "w") as f:
        json.dump(explanations, f, indent=2)

    print(f"  ✅ AI explanations saved → {path}")
    return path


def run_explainer() -> dict:
    """
    Public function called by run_monitor.py.
    Loads latest reports, calls AI, saves and returns explanations.
    """
    print("\n📂 Loading latest reports...")

    profile    = load_latest_report("profile_")
    ge_summary = load_latest_report("ge_results_")

    print("\n🤖 Building context summary...")
    context_summary = build_context_summary(profile, ge_summary)

    print("\n🤖 Calling AI API...")
    prompt       = build_prompt(context_summary)
    explanations = call_ai_api(prompt)

    print_explanations(explanations)
    save_explanations(explanations)

    return explanations


# ── ENTRY POINT ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    run_explainer()