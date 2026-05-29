"""Plot occupation-level robot exposure against 2024-2034 annual openings."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROBOT_EXPOSURE_PATH = Path("data/exposure_by_occ1990dd_lswt2010.xls")
SOC_CROSSWALK_PATH = Path("data/soc_occ_crosswalk.xlsx")
EMPLOYMENT_PROJECTIONS_PATH = Path("data/Employment Projections.csv")

ROBOT_VS_OPENINGS_TABLE_PATH = Path(
    "outputs/tables/robot_exposure_vs_annual_openings_by_occupation.csv"
)
ROBOT_VS_OPENINGS_FIGURE_PNG = Path(
    "outputs/figures/robot_exposure_vs_annual_openings_scatter.png"
)
ROBOT_VS_OPENINGS_FIGURE_SVG = Path(
    "outputs/figures/robot_exposure_vs_annual_openings_scatter.svg"
)


def show_table(df):
    try:
        display(df)
    except NameError:
        print(df.to_string(index=False))


def numeric_series(df, column):
    series = df[column]
    if pd.api.types.is_numeric_dtype(series):
        return series
    return pd.to_numeric(series, errors="coerce")


def ensure_weight_column(df, weight_col):
    out = df.copy()
    if weight_col not in out.columns:
        out[weight_col] = pd.to_numeric(out.get("PERWT", 0), errors="coerce").fillna(0)
    else:
        out[weight_col] = pd.to_numeric(out[weight_col], errors="coerce").fillna(0)
    return out


def find_occ1990dd_column(df):
    for column in ["OCC1990DD", "occ1990dd", "OCC1990", "occ1990"]:
        if column in df.columns:
            return column
    return None


if "ipums_df" not in globals():
    raise RuntimeError("Run the ACS data cells before this section.")

if "WEIGHT_COL" not in globals():
    WEIGHT_COL = "PERWT_num"

ipums_df = ensure_weight_column(ipums_df, WEIGHT_COL)
acs_occ_col = next((col for col in ["OCC", "OCC2010"] if col in ipums_df.columns), None)
if acs_occ_col is None:
    raise KeyError("Need OCC or OCC2010 in ipums_df.")

robot_occ_col = find_occ1990dd_column(ipums_df)
if robot_occ_col is None:
    available_occ_cols = sorted([col for col in ipums_df.columns if "OCC" in col.upper()])
    raise KeyError(
        "Need OCC1990DD or OCC1990 in ipums_df to map robot exposure scores. "
        f"Available occupation columns: {available_occ_cols}"
    )

workers = ipums_df.copy()
if "EMPSTAT" in workers.columns:
    workers = workers[numeric_series(workers, "EMPSTAT") == 1].copy()
workers = workers[workers[WEIGHT_COL] > 0].copy()

workers["acs_occ_code"] = pd.to_numeric(workers[acs_occ_col], errors="coerce").astype("Int64")
workers["robot_occ1990dd_code"] = pd.to_numeric(workers[robot_occ_col], errors="coerce").astype("Int64")

robot_exposure_raw = pd.read_csv(ROBOT_EXPOSURE_PATH)
robot_exposure_scores = (
    robot_exposure_raw
    .rename(columns={"occ1990dd": "robot_occ1990dd_code", "pct_robot": "pct_robot"})
    [["robot_occ1990dd_code", "occ1990dd_title", "pct_robot"]]
    .copy()
)
robot_exposure_scores["robot_occ1990dd_code"] = pd.to_numeric(
    robot_exposure_scores["robot_occ1990dd_code"],
    errors="coerce",
).astype("Int64")
robot_exposure_scores["pct_robot"] = pd.to_numeric(
    robot_exposure_scores["pct_robot"],
    errors="coerce",
)
robot_exposure_scores["occ1990dd_title"] = (
    robot_exposure_scores["occ1990dd_title"].astype(str).str.strip()
)
robot_exposure_scores = robot_exposure_scores.dropna(
    subset=["robot_occ1990dd_code", "pct_robot"]
)

robot_lookup = (
    robot_exposure_scores
    .groupby("robot_occ1990dd_code", as_index=True)
    .agg(
        pct_robot=("pct_robot", "mean"),
        occ1990dd_title=("occ1990dd_title", "first"),
    )
)

workers["pct_robot"] = workers["robot_occ1990dd_code"].map(robot_lookup["pct_robot"])
workers_with_scores = workers[workers["pct_robot"].notna() & workers["acs_occ_code"].notna()].copy()

acs_robot_summary = (
    workers_with_scores
    .groupby("acs_occ_code", as_index=False)
    .apply(
        lambda g: pd.Series({
            "weighted_mean_pct_robot": np.average(g["pct_robot"], weights=g[WEIGHT_COL]),
            "weighted_workers_with_robot_score": g[WEIGHT_COL].sum(),
            "unweighted_workers_with_robot_score": len(g),
        }),
        include_groups=False,
    )
)
acs_robot_summary["acs_occ_code"] = acs_robot_summary["acs_occ_code"].astype("Int64")

soc_crosswalk_raw = pd.read_excel(
    SOC_CROSSWALK_PATH,
    sheet_name="NEM SOC ACS crosswalk",
    header=4,
)
soc_crosswalk = (
    soc_crosswalk_raw
    .rename(columns={
        "ACS code": "acs_occ_code",
        "ACS cccupational title": "acs_occupation_title",
        "National Employment Matrix code": "soc_code",
    })
    [["acs_occ_code", "acs_occupation_title", "soc_code"]]
    .dropna(subset=["acs_occ_code", "soc_code"])
    .copy()
)
soc_crosswalk["acs_occ_code"] = pd.to_numeric(soc_crosswalk["acs_occ_code"], errors="coerce").astype("Int64")
soc_crosswalk["soc_code"] = soc_crosswalk["soc_code"].astype(str).str.strip()
soc_crosswalk["acs_occupation_title"] = soc_crosswalk["acs_occupation_title"].astype(str).str.strip()

projections_raw = pd.read_csv(EMPLOYMENT_PROJECTIONS_PATH)
projections = (
    projections_raw
    .rename(columns={
        "Occupation Code": "soc_code",
        "Occupation Title": "soc_occupation_title",
        "Occupational Openings, 2024-2034 Annual Average": "annual_openings_2024_2034",
    })
    [["soc_code", "soc_occupation_title", "annual_openings_2024_2034"]]
    .dropna(subset=["soc_code", "annual_openings_2024_2034"])
    .copy()
)
projections["soc_code"] = projections["soc_code"].astype(str).str.strip()
projections["annual_openings_2024_2034"] = pd.to_numeric(
    projections["annual_openings_2024_2034"],
    errors="coerce",
)
projections = projections.dropna(subset=["annual_openings_2024_2034"])

crosswalk_with_openings = soc_crosswalk.merge(projections, on="soc_code", how="inner")
acs_openings_summary = (
    crosswalk_with_openings
    .groupby("acs_occ_code", as_index=False)
    .agg(
        annual_openings_2024_2034=("annual_openings_2024_2034", "sum"),
        linked_soc_count=("soc_code", "nunique"),
        acs_occupation_title=("acs_occupation_title", "first"),
    )
)

robot_vs_openings = acs_robot_summary.merge(acs_openings_summary, on="acs_occ_code", how="inner")
robot_vs_openings = robot_vs_openings.dropna(
    subset=["weighted_mean_pct_robot", "annual_openings_2024_2034"]
)
robot_vs_openings = robot_vs_openings.sort_values(
    ["annual_openings_2024_2034", "weighted_mean_pct_robot"],
    ascending=[False, False],
).reset_index(drop=True)

ROBOT_VS_OPENINGS_TABLE_PATH.parent.mkdir(parents=True, exist_ok=True)
robot_vs_openings.to_csv(ROBOT_VS_OPENINGS_TABLE_PATH, index=False)

point_color = globals().get("VALDOS_COLORS", {}).get("primary", "#0b7285")
highlight_color = globals().get("VALDOS_COLORS", {}).get("secondary", "#c44536")

fig, ax = plt.subplots(figsize=(12, 8))
ax.scatter(
    robot_vs_openings["weighted_mean_pct_robot"],
    robot_vs_openings["annual_openings_2024_2034"],
    s=28,
    alpha=0.5,
    color=point_color,
    edgecolor="none",
)

top_openings = robot_vs_openings.nlargest(10, "annual_openings_2024_2034")
for _, row in top_openings.iterrows():
    ax.scatter(
        row["weighted_mean_pct_robot"],
        row["annual_openings_2024_2034"],
        s=40,
        color=highlight_color,
        alpha=0.9,
        edgecolor="white",
        linewidth=0.4,
    )
    label_text = str(row["acs_occupation_title"])[:45]
    ax.annotate(
        label_text,
        (row["weighted_mean_pct_robot"], row["annual_openings_2024_2034"]),
        textcoords="offset points",
        xytext=(4, 4),
        fontsize=8,
        alpha=0.85,
    )

corr = robot_vs_openings["weighted_mean_pct_robot"].corr(
    robot_vs_openings["annual_openings_2024_2034"]
)
ax.set_title(
    "Robot Exposure vs Occupational Openings (ACS-Mapped Occupations)",
    fontsize=15,
    fontweight="bold",
)
ax.set_xlabel("Weighted mean robot exposure score (pct_robot)", fontsize=12)
ax.set_ylabel("Annual openings, 2024-2034 average", fontsize=12)
ax.grid(alpha=0.25)
if pd.notna(corr):
    ax.text(
        0.01,
        0.98,
        f"Correlation: {corr:.2f}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=10,
        bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"},
    )

plt.tight_layout()

ROBOT_VS_OPENINGS_FIGURE_PNG.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(ROBOT_VS_OPENINGS_FIGURE_PNG, dpi=300, bbox_inches="tight")
fig.savefig(ROBOT_VS_OPENINGS_FIGURE_SVG, bbox_inches="tight")
plt.show()

print(f"Saved occupation-level merge to {ROBOT_VS_OPENINGS_TABLE_PATH}")
print(f"Saved figure to {ROBOT_VS_OPENINGS_FIGURE_PNG}")
print(f"Saved figure to {ROBOT_VS_OPENINGS_FIGURE_SVG}")
print(
    "Note: annual openings are aggregated from SOC to ACS using the crosswalk and "
    "summed within each ACS occupation code."
)

show_table(robot_vs_openings.head(20))