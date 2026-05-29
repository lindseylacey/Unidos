"""Map pct_robot exposure scores to workers and summarize by status."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROBOT_EXPOSURE_PATH = Path("data/exposure_by_occ1990dd_lswt2010.xls")
ROBOT_EXPOSURE_SCORE_COL = "pct_robot"
ROBOT_OUTPUT_PATH = Path("outputs/tables/robot_exposure_pct_robot_by_worker_status_summary.csv")
ROBOT_FIGURE_DIR = Path("outputs/figures")


def show_table(df):
    try:
        display(df)
    except NameError:
        print(df.to_string(index=False))


def ensure_weight_column(df, weight_col):
    if weight_col not in df.columns:
        df[weight_col] = pd.to_numeric(df.get("PERWT", 0), errors="coerce").fillna(0)
    else:
        df[weight_col] = pd.to_numeric(df[weight_col], errors="coerce").fillna(0)
    return df


def employed_workers(df, weight_col):
    out = df.copy()
    out = ensure_weight_column(out, weight_col)

    if "EMPSTAT" in out.columns:
        out = out[pd.to_numeric(out["EMPSTAT"], errors="coerce") == 1].copy()

    return out[out[weight_col] > 0].copy()


def weighted_gaussian_kde(score_weights, grid):
    values = score_weights.index.to_numpy(dtype=float)
    weights = score_weights.to_numpy(dtype=float)
    valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    values = values[valid]
    weights = weights[valid]

    if len(values) < 2 or weights.sum() <= 0:
        return np.full_like(grid, np.nan, dtype=float)

    weight_sum = weights.sum()
    weighted_mean = np.average(values, weights=weights)
    weighted_var = np.average((values - weighted_mean) ** 2, weights=weights)
    weighted_sd = np.sqrt(weighted_var)

    if not np.isfinite(weighted_sd) or weighted_sd <= 0:
        return np.full_like(grid, np.nan, dtype=float)

    effective_n = weight_sum ** 2 / np.square(weights).sum()
    bandwidth = 1.06 * weighted_sd * effective_n ** (-1 / 5)
    if not np.isfinite(bandwidth) or bandwidth <= 0:
        bandwidth = weighted_sd * 0.25

    z = (grid[:, None] - values[None, :]) / bandwidth
    density = np.exp(-0.5 * z**2) @ weights
    density = density / (weight_sum * bandwidth * np.sqrt(2 * np.pi))
    return density


def find_occ1990dd_column(df):
    for column in ["OCC1990DD", "occ1990dd", "OCC1990", "occ1990"]:
        if column in df.columns:
            return column
    return None


def attach_robot_exposure(df, occ_col, lookup):
    out = df.copy()
    out["robot_occ1990dd_code"] = pd.to_numeric(
        out[occ_col],
        errors="coerce",
    ).astype("Int64")
    out["pct_robot"] = out["robot_occ1990dd_code"].map(lookup["pct_robot"])
    out["pct_robot_occupation_title"] = out["robot_occ1990dd_code"].map(
        lookup["occ1990dd_title"]
    )
    return out


if "ipums_df" not in globals() or "foreign_born_df" not in globals():
    raise RuntimeError("Run the ACS data and likely-legal classification cells before this section.")

if "WEIGHT_COL" not in globals():
    WEIGHT_COL = "PERWT_num"

if "likely_legal" not in foreign_born_df.columns:
    raise RuntimeError("Run the likely-legal classification cells before this section.")

ipums_df = ensure_weight_column(ipums_df, WEIGHT_COL)
foreign_born_df = ensure_weight_column(foreign_born_df, WEIGHT_COL)

# This file has an .xls extension, but its contents are comma-delimited text.
robot_exposure_raw = pd.read_csv(ROBOT_EXPOSURE_PATH)
robot_exposure_scores = (
    robot_exposure_raw
    .rename(columns={
        "occ1990dd": "occ1990dd",
        "occ1990dd_title": "occ1990dd_title",
        ROBOT_EXPOSURE_SCORE_COL: "pct_robot",
    })
    .dropna(subset=["occ1990dd", "pct_robot"])
    .copy()
)
robot_exposure_scores["occ1990dd"] = pd.to_numeric(
    robot_exposure_scores["occ1990dd"],
    errors="coerce",
).astype("Int64")
robot_exposure_scores["pct_robot"] = pd.to_numeric(
    robot_exposure_scores["pct_robot"],
    errors="coerce",
)
robot_exposure_scores["occ1990dd_title"] = (
    robot_exposure_scores["occ1990dd_title"].astype(str).str.strip()
)
robot_exposure_scores = robot_exposure_scores.dropna(subset=["occ1990dd", "pct_robot"])

robot_exposure_lookup = (
    robot_exposure_scores
    .groupby("occ1990dd", as_index=True)
    .agg(
        occ1990dd_title=("occ1990dd_title", "first"),
        pct_robot=("pct_robot", "mean"),
    )
)

ipums_robot_occ_col = find_occ1990dd_column(ipums_df)
foreign_robot_occ_col = find_occ1990dd_column(foreign_born_df)

if ipums_robot_occ_col is None or foreign_robot_occ_col is None:
    available_occ_cols = sorted([col for col in ipums_df.columns if "OCC" in col.upper()])
    print(
        f"Loaded {len(robot_exposure_scores):,} pct_robot exposure scores from "
        f"{ROBOT_EXPOSURE_PATH}."
    )
    print("These scores are keyed by occ1990dd.")
    print(f"Current ACS extract occupation columns: {available_occ_cols}")
    raise KeyError(
        "The current ACS extract does not include OCC1990DD/OCC1990, so pct_robot "
        "cannot be mapped yet. Add OCC1990/OCC1990DD to the IPUMS extract or provide "
        "a crosswalk from OCC/OCC2010 to occ1990dd."
    )

if ipums_robot_occ_col.upper() == "OCC1990" or foreign_robot_occ_col.upper() == "OCC1990":
    print(
        "Using OCC1990 as the occ1990dd key for pct_robot. Please verify this matches "
        "the exposure file's occupation coding."
    )

ipums_df = attach_robot_exposure(ipums_df, ipums_robot_occ_col, robot_exposure_lookup)
foreign_born_df = attach_robot_exposure(
    foreign_born_df,
    foreign_robot_occ_col,
    robot_exposure_lookup,
)

df_likely_legal = foreign_born_df[
    foreign_born_df["likely_legal"].fillna(False).astype(bool)
].copy()
df_likely_unauthorized = foreign_born_df[
    ~foreign_born_df["likely_legal"].fillna(False).astype(bool)
].copy()
native_born_df = ipums_df[pd.to_numeric(ipums_df["BPL"], errors="coerce") < 150].copy()

worker_status_groups = {
    "Likely legal": df_likely_legal,
    "Likely undocumented": df_likely_unauthorized,
    "Native-born": native_born_df,
}

score_distributions = {}
summary_rows = []
for label, source_df in worker_status_groups.items():
    workers = employed_workers(source_df, WEIGHT_COL)
    workers_with_scores = workers[workers["pct_robot"].notna()].copy()
    score_weights = (
        workers_with_scores
        .groupby("pct_robot", dropna=True)[WEIGHT_COL]
        .sum()
        .sort_index()
    )
    score_distributions[label] = score_weights

    weighted_workers = workers[WEIGHT_COL].sum()
    weighted_workers_with_scores = workers_with_scores[WEIGHT_COL].sum()
    if len(score_weights) > 0 and weighted_workers_with_scores > 0:
        weighted_mean_pct_robot = np.average(
            score_weights.index.to_numpy(dtype=float),
            weights=score_weights.to_numpy(dtype=float),
        )
    else:
        weighted_mean_pct_robot = np.nan

    summary_rows.append({
        "worker_status": label,
        "unweighted_workers_with_score": len(workers_with_scores),
        "weighted_workers": weighted_workers,
        "weighted_workers_with_score": weighted_workers_with_scores,
        "score_coverage_pct": (
            weighted_workers_with_scores / weighted_workers * 100
            if weighted_workers
            else np.nan
        ),
        "weighted_mean_pct_robot": weighted_mean_pct_robot,
    })

robot_exposure_status_summary = pd.DataFrame(summary_rows)

ROBOT_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
robot_exposure_status_summary.to_csv(ROBOT_OUTPUT_PATH, index=False)

score_value_arrays = [
    score_weights.index.to_numpy(dtype=float)
    for score_weights in score_distributions.values()
    if len(score_weights) > 0
]
if not score_value_arrays:
    raise ValueError("No employed workers have mapped pct_robot exposure scores.")
all_score_values = np.concatenate(score_value_arrays)

x_min = np.nanmin(all_score_values)
x_max = np.nanmax(all_score_values)
x_padding = max((x_max - x_min) * 0.05, 0.01)
x_grid = np.linspace(x_min - x_padding, x_max + x_padding, 500)

status_colors = {
    "Likely legal": "#087E8B",
    "Likely undocumented": "#D95F02",
    "Native-born": "#4B5563",
}

fig, ax = plt.subplots(figsize=(11, 7))
for label, score_weights in score_distributions.items():
    density = weighted_gaussian_kde(score_weights, x_grid)
    color = status_colors[label]
    ax.plot(x_grid, density, label=label, color=color, linewidth=2.5)
    ax.fill_between(x_grid, density, color=color, alpha=0.12)

    if len(score_weights) > 0:
        mean_score = np.average(
            score_weights.index.to_numpy(dtype=float),
            weights=score_weights.to_numpy(dtype=float),
        )
        ax.axvline(mean_score, color=color, linestyle="--", linewidth=1.2, alpha=0.6)

ax.set_title(
    "Robot Exposure Score Distribution by Worker Status",
    fontsize=15,
    fontweight="bold",
)
ax.set_xlabel("Robot exposure score", fontsize=12)
ax.set_ylabel("Weighted density", fontsize=12)
ax.grid(axis="y", alpha=0.25)
ax.legend(title="Worker status", frameon=False)

plt.tight_layout()

ROBOT_FIGURE_DIR.mkdir(parents=True, exist_ok=True)
robot_exposure_kde_png = ROBOT_FIGURE_DIR / "robot_exposure_kde_by_worker_status.png"
robot_exposure_kde_svg = ROBOT_FIGURE_DIR / "robot_exposure_kde_by_worker_status.svg"
fig.savefig(robot_exposure_kde_png, dpi=300, bbox_inches="tight")
fig.savefig(robot_exposure_kde_svg, bbox_inches="tight")
plt.show()

print(
    f"Loaded {len(robot_exposure_scores):,} pct_robot exposure scores from "
    f"{ROBOT_EXPOSURE_PATH}."
)
print(f"Saved robot exposure status summary to {ROBOT_OUTPUT_PATH}.")
print(f"Saved figure to {robot_exposure_kde_png} and {robot_exposure_kde_svg}.")
show_table(robot_exposure_status_summary)
