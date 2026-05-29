"""Print selected median wages by nativity/documentation status and geography."""

from pathlib import Path

import pandas as pd


SELECTED_WAGE_OCCUPATIONS = {
    6040: "Graders and sorters, agricultural products",
    4230: "Maids and housekeeping cleaners",
    6330: "Drywall installers, ceiling tile installers, and tapers",
}
SELECTED_WAGE_STATES = {
    12: "FL",
    48: "TX",
    6: "CA",
}
STATUS_ORDER = ["Native born", "Documented", "Undocumented"]
WAGE_COL = "INCWAGE"
FULL_TIME_MIN_HOURS = 35
MIN_WEEKS_WORKED = 50
FULL_TIME_HOURS_CANDIDATES = ["UHRSWORK", "UHRSWORKLY"]
MERGE_KEYS = ["SAMPLE", "SERIAL", "PERNUM"]
NATIVE_BORN_BPL_THRESHOLD = 150


def numeric_series(df, column):
    series = df[column]
    if pd.api.types.is_numeric_dtype(series):
        return series
    return pd.to_numeric(series, errors="coerce")


def weighted_median(values, weights):
    tmp = pd.DataFrame({"value": values, "weight": weights}).dropna()
    tmp = tmp[tmp["weight"] > 0].sort_values("value")
    if tmp.empty:
        return float("nan")
    cutoff = tmp["weight"].sum() / 2
    return float(tmp.loc[tmp["weight"].cumsum() >= cutoff, "value"].iloc[0])


if "ipums_df" not in globals():
    raise RuntimeError("Run the ACS data cells first.")
if "foreign_born_df" not in globals():
    raise RuntimeError("Run the likely-legal classification cells first.")
if "likely_legal" not in foreign_born_df.columns:
    raise RuntimeError("Run the likely-legal classification cells first.")

if "WEIGHT_COL" not in globals():
    WEIGHT_COL = "PERWT_num"

analysis_df = ipums_df.copy()
foreign_born_lookup = foreign_born_df[MERGE_KEYS + ["likely_legal"]].drop_duplicates(MERGE_KEYS)
analysis_df = analysis_df.merge(foreign_born_lookup, on=MERGE_KEYS, how="left")

if WEIGHT_COL not in analysis_df.columns:
    analysis_df[WEIGHT_COL] = pd.to_numeric(analysis_df.get("PERWT", 0), errors="coerce").fillna(0)
if WEIGHT_COL not in foreign_born_df.columns:
    foreign_born_df[WEIGHT_COL] = pd.to_numeric(foreign_born_df.get("PERWT", 0), errors="coerce").fillna(0)

occ_col = next((col for col in ["OCC", "OCC2010"] if col in analysis_df.columns), None)
if occ_col is None:
    raise KeyError("Need OCC or OCC2010 in ipums_df.")

full_time_hours_col = next(
    (col for col in FULL_TIME_HOURS_CANDIDATES if col in analysis_df.columns),
    None,
)
if full_time_hours_col is None:
    available_work_cols = [
        col
        for col in analysis_df.columns
        if any(token in col.upper() for token in ["HRS", "HOUR", "WEEK", "WKSWORK"])
    ]
    raise KeyError(
        "Cannot restrict selected wage medians to full-time workers because this "
        "ACS extract does not include a usual-hours variable. Add UHRSWORK to the "
        "IPUMS extract and rerun the notebook. Available hours/weeks-like columns: "
        f"{available_work_cols or 'none'}"
    )

required_cols = {"STATEFIP", WAGE_COL, WEIGHT_COL, full_time_hours_col, "WKSWORK1", "BPL", "likely_legal"}
missing_cols = required_cols.difference(analysis_df.columns)
if missing_cols:
    raise KeyError(f"Missing columns needed from ipums_df: {sorted(missing_cols)}")

statefip = numeric_series(analysis_df, "STATEFIP")
occ_code = numeric_series(analysis_df, occ_col)
annual_wage = numeric_series(analysis_df, WAGE_COL)
hours = numeric_series(analysis_df, full_time_hours_col)
weeks_worked = numeric_series(analysis_df, "WKSWORK1")
weights = numeric_series(analysis_df, WEIGHT_COL).fillna(0)
likely_legal = analysis_df["likely_legal"].fillna(False).astype(bool)
bpl = numeric_series(analysis_df, "BPL")

base_valid_mask = (
    annual_wage.notna()
    & (annual_wage > 0)
    & (annual_wage < 999998)
    & (hours >= FULL_TIME_MIN_HOURS)
    & (weeks_worked >= MIN_WEEKS_WORKED)
    & (weights > 0)
)
if "EMPSTAT" in analysis_df.columns:
    base_valid_mask &= numeric_series(analysis_df, "EMPSTAT") == 1
if "AGE" in analysis_df.columns:
    base_valid_mask &= numeric_series(analysis_df, "AGE").between(18, 64, inclusive="both")

status_specs = [
    ("Native born", bpl < NATIVE_BORN_BPL_THRESHOLD),
    ("Documented", (bpl >= NATIVE_BORN_BPL_THRESHOLD) & likely_legal),
    ("Undocumented", (bpl >= NATIVE_BORN_BPL_THRESHOLD) & ~likely_legal),
]

geography_specs = [
    ("National", pd.Series(True, index=analysis_df.index), 0),
    ("FL", statefip == 12, 12),
    ("TX", statefip == 48, 48),
    ("CA", statefip == 6, 6),
]

median_rows = []
for status_name, status_mask in status_specs:
    for geography_name, geography_mask, geography_statefip in geography_specs:
        for acs_occ_code, occupation_name in SELECTED_WAGE_OCCUPATIONS.items():
            current_mask = (
                status_mask
                & geography_mask
                & base_valid_mask
                & (occ_code == acs_occ_code)
            )
            group = analysis_df.loc[current_mask]
            median_rows.append({
                "status": status_name,
                "geography": geography_name,
                "statefip": geography_statefip,
                "acs_occ_code": acs_occ_code,
                "occupation_name": occupation_name,
                "weighted_median_annual_wage": weighted_median(group[WAGE_COL], group[WEIGHT_COL]) if not group.empty else float("nan"),
                "weighted_workers_with_wage": group[WEIGHT_COL].sum(),
                "unweighted_workers_with_wage": len(group),
            })

status_median_wages = pd.DataFrame(median_rows)
status_median_wages["status"] = pd.Categorical(
    status_median_wages["status"],
    categories=STATUS_ORDER,
    ordered=True,
)
status_median_wages["geography"] = pd.Categorical(
    status_median_wages["geography"],
    categories=["National", "FL", "TX", "CA"],
    ordered=True,
)
status_median_wages = status_median_wages.sort_values(
    ["status", "geography", "acs_occ_code"],
    kind="stable",
).reset_index(drop=True)

output_dir = Path("outputs") / "tables"
output_dir.mkdir(parents=True, exist_ok=True)
status_median_wages_csv = Path(
    globals().get(
        "OUTPUT_CSV_PATH",
        output_dir / "selected_median_wages_by_status_fl_tx_ca_nat.csv",
    )
)
write_output_csv = bool(globals().get("WRITE_OUTPUT_CSV", True))
if write_output_csv:
    try:
        status_median_wages.to_csv(status_median_wages_csv, index=False)
    except PermissionError:
        status_median_wages_csv = output_dir / "selected_median_wages_by_status_fl_tx_ca_nat_rerun.csv"
        status_median_wages.to_csv(status_median_wages_csv, index=False)


def format_currency(value):
    return "" if pd.isna(value) else f"${value:,.0f}"


def format_count(value):
    return "" if pd.isna(value) else f"{value:,.0f}"


print(
    "Selected median annual wages for native born, documented, and undocumented "
    f"workers in FL, TX, CA, and nationally ({full_time_hours_col} >= {FULL_TIME_MIN_HOURS} "
    f"hours/week, WKSWORK1 >= {MIN_WEEKS_WORKED} weeks worked):"
)
print(f"Filtered to {len(analysis_df.loc[base_valid_mask]):,} unweighted matching records across all statuses.")
if write_output_csv:
    print(f"Saved table to {status_median_wages_csv}")
else:
    print(f"Skipped CSV write; output path would have been {status_median_wages_csv}")
print(
    status_median_wages.to_string(
        index=False,
        formatters={
            "weighted_median_annual_wage": format_currency,
            "weighted_workers_with_wage": format_count,
            "unweighted_workers_with_wage": format_count,
        },
    )
)