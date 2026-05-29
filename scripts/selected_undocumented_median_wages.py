"""Print selected full-time likely undocumented median wages by state."""

from pathlib import Path

import pandas as pd


SELECTED_UNDOC_WAGE_OCCUPATIONS = {
    6040: "Graders and sorters, agricultural products",
    4230: "Maids and housekeeping cleaners",
    6330: "Drywall installers, ceiling tile installers, and tapers",
}
SELECTED_UNDOC_WAGE_STATES = {
    12: "FL",
    48: "TX",
    6: "CA",
}
WAGE_COL = "INCWAGE"
FULL_TIME_MIN_HOURS = 35
FULL_TIME_HOURS_CANDIDATES = ["UHRSWORK", "UHRSWORKLY"]
MIN_WEEKS_WORKED = 50
WEEKS_WORKED_COL = "WKSWORK1"


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


if "foreign_born_df" not in globals():
    raise RuntimeError("Run the ACS data and likely-legal classification cells first.")
if "likely_legal" not in foreign_born_df.columns:
    raise RuntimeError("Run the likely-legal classification cells first.")

if "WEIGHT_COL" not in globals():
    WEIGHT_COL = "PERWT_num"
if WEIGHT_COL not in foreign_born_df.columns:
    foreign_born_df[WEIGHT_COL] = pd.to_numeric(
        foreign_born_df.get("PERWT", 0),
        errors="coerce",
    ).fillna(0)

occ_col = next((col for col in ["OCC", "OCC2010"] if col in foreign_born_df.columns), None)
if occ_col is None:
    raise KeyError("Need OCC or OCC2010 in foreign_born_df.")

full_time_hours_col = next(
    (col for col in FULL_TIME_HOURS_CANDIDATES if col in foreign_born_df.columns),
    None,
)
if full_time_hours_col is None:
    available_work_cols = [
        col
        for col in foreign_born_df.columns
        if any(token in col.upper() for token in ["HRS", "HOUR", "WEEK", "WKSWORK"])
    ]
    raise KeyError(
        "Cannot restrict selected wage medians to full-time workers because this "
        "ACS extract does not include a usual-hours variable. Add UHRSWORK to the "
        "IPUMS extract and rerun the notebook. Available hours/weeks-like columns: "
        f"{available_work_cols or 'none'}"
    )

required_cols = {"STATEFIP", WAGE_COL, "likely_legal", WEIGHT_COL, full_time_hours_col}
required_cols.add(WEEKS_WORKED_COL)
missing_cols = required_cols.difference(foreign_born_df.columns)
if missing_cols:
    raise KeyError(f"Missing columns needed from foreign_born_df: {sorted(missing_cols)}")

base = foreign_born_df
statefip = numeric_series(base, "STATEFIP")
occ_code = numeric_series(base, occ_col)
annual_wage = numeric_series(base, WAGE_COL)
hours = numeric_series(base, full_time_hours_col)
weeks_worked = numeric_series(base, WEEKS_WORKED_COL)
weights = numeric_series(base, WEIGHT_COL).fillna(0)
likely_legal = base["likely_legal"].fillna(False).astype(bool)

valid_wage_mask = (
    ~likely_legal
    & statefip.isin(SELECTED_UNDOC_WAGE_STATES)
    & occ_code.isin(SELECTED_UNDOC_WAGE_OCCUPATIONS)
    & annual_wage.notna()
    & (annual_wage > 0)
    & (annual_wage < 999998)
    & (hours >= FULL_TIME_MIN_HOURS)
    & (weeks_worked >= MIN_WEEKS_WORKED)
    & (weights > 0)
)
if "EMPSTAT" in base.columns:
    valid_wage_mask &= numeric_series(base, "EMPSTAT") == 1
if "AGE" in base.columns:
    valid_wage_mask &= numeric_series(base, "AGE").between(18, 64, inclusive="both")

selected_wage_inputs = pd.DataFrame({
    "statefip": statefip[valid_wage_mask].astype(int),
    "acs_occ_code": occ_code[valid_wage_mask].astype(int),
    "annual_wage": annual_wage[valid_wage_mask],
    WEIGHT_COL: weights[valid_wage_mask],
})

median_rows = []
for (statefip_value, acs_occ_code), group in selected_wage_inputs.groupby(
    ["statefip", "acs_occ_code"],
    dropna=False,
):
    median_rows.append({
        "statefip": int(statefip_value),
        "acs_occ_code": int(acs_occ_code),
        "weighted_median_annual_wage": weighted_median(
            group["annual_wage"],
            group[WEIGHT_COL],
        ),
        "weighted_workers_with_wage": group[WEIGHT_COL].sum(),
        "unweighted_workers_with_wage": len(group),
    })

state_occ_index = pd.MultiIndex.from_product(
    [
        list(SELECTED_UNDOC_WAGE_STATES),
        list(SELECTED_UNDOC_WAGE_OCCUPATIONS),
    ],
    names=["statefip", "acs_occ_code"],
).to_frame(index=False)
median_summary = pd.DataFrame(median_rows)
if median_summary.empty:
    median_summary = pd.DataFrame(columns=[
        "statefip",
        "acs_occ_code",
        "weighted_median_annual_wage",
        "weighted_workers_with_wage",
        "unweighted_workers_with_wage",
    ])

selected_undoc_median_wages = state_occ_index.merge(
    median_summary,
    on=["statefip", "acs_occ_code"],
    how="left",
)
selected_undoc_median_wages["state"] = selected_undoc_median_wages["statefip"].map(
    SELECTED_UNDOC_WAGE_STATES
)
selected_undoc_median_wages["occupation_name"] = selected_undoc_median_wages[
    "acs_occ_code"
].map(SELECTED_UNDOC_WAGE_OCCUPATIONS)
selected_undoc_median_wages = selected_undoc_median_wages[[
    "state",
    "statefip",
    "acs_occ_code",
    "occupation_name",
    "weighted_median_annual_wage",
    "weighted_workers_with_wage",
    "unweighted_workers_with_wage",
]]
selected_undoc_median_wages["weighted_median_annual_wage"] = (
    selected_undoc_median_wages["weighted_median_annual_wage"].round(0)
)
selected_undoc_median_wages["weighted_workers_with_wage"] = (
    selected_undoc_median_wages["weighted_workers_with_wage"].round(0)
)

output_dir = Path("outputs") / "tables"
output_dir.mkdir(parents=True, exist_ok=True)
selected_undoc_median_wages_csv = Path(
    globals().get(
        "OUTPUT_CSV_PATH",
        output_dir / "median_wages.csv",
    )
)
write_output_csv = bool(globals().get("WRITE_OUTPUT_CSV", True))
if write_output_csv:
    try:
        selected_undoc_median_wages.to_csv(selected_undoc_median_wages_csv, index=False)
    except PermissionError:
        selected_undoc_median_wages_csv = output_dir / "median_wages_rerun.csv"
        selected_undoc_median_wages.to_csv(selected_undoc_median_wages_csv, index=False)


def format_currency(value):
    return "" if pd.isna(value) else f"${value:,.0f}"


def format_count(value):
    return "" if pd.isna(value) else f"{value:,.0f}"


print(
    "Weighted median annual wages for likely undocumented full-time workers "
    f"in FL, TX, and CA ({full_time_hours_col} >= {FULL_TIME_MIN_HOURS} hours/week, "
    f"{WEEKS_WORKED_COL} >= {MIN_WEEKS_WORKED} weeks worked):"
)
print(f"Filtered to {len(selected_wage_inputs):,} unweighted matching records.")
if write_output_csv:
    print(f"Saved table to {selected_undoc_median_wages_csv}")
else:
    print(f"Skipped CSV write; output path would have been {selected_undoc_median_wages_csv}")
print(
    selected_undoc_median_wages.to_string(
        index=False,
        formatters={
            "weighted_median_annual_wage": format_currency,
            "weighted_workers_with_wage": format_count,
            "unweighted_workers_with_wage": format_count,
        },
    )
)
