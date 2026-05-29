"""Map AI exposure scores to ACS occupations and plot worker-status KDEs."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


AI_EXPOSURE_PATH = Path("data/exposure_scores.xlsx")
AI_EXPOSURE_SHEET = "F1"  # SOC-level PCA weighted scores


def show_table(df):
    try:
        display(df)
    except NameError:
        print(df.to_string(index=False))


def join_unique_text(values):
    unique_values = pd.Series(values).dropna().astype(str).str.strip()
    unique_values = unique_values[unique_values.ne("")].drop_duplicates().sort_values()
    return "; ".join(unique_values)


def ensure_weight_column(df, weight_col):
    if weight_col not in df.columns:
        df[weight_col] = pd.to_numeric(df.get("PERWT", 0), errors="coerce").fillna(0)
    else:
        df[weight_col] = pd.to_numeric(df[weight_col], errors="coerce").fillna(0)
    return df


def attach_ai_exposure(df, lookup):
    occ_source_col = "OCC" if "OCC" in df.columns else "OCC2010"
    out = df.copy()
    out["acs_occ_code"] = pd.to_numeric(out[occ_source_col], errors="coerce").astype("Int64")

    for column in [
        "ai_exposure_score",
        "ai_exposure_score_min",
        "ai_exposure_score_max",
        "ai_exposure_matched_soc_codes",
        "ai_exposure_total_soc_codes",
        "ai_exposure_occupation_title",
    ]:
        out[column] = out["acs_occ_code"].map(lookup[column])

    return out


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


if "ipums_df" not in globals() or "foreign_born_df" not in globals():
    raise RuntimeError("Run the ACS data and likely-legal classification cells before this section.")

if "WEIGHT_COL" not in globals():
    WEIGHT_COL = "PERWT_num"

ipums_df = ensure_weight_column(ipums_df, WEIGHT_COL)
foreign_born_df = ensure_weight_column(foreign_born_df, WEIGHT_COL)

ai_exposure_raw = pd.read_excel(
    AI_EXPOSURE_PATH,
    sheet_name=AI_EXPOSURE_SHEET,
    header=5,
)

ai_exposure_scores = (
    ai_exposure_raw
    .rename(columns={
        "SOC2018": "soc_code",
        "Occupation": "ai_exposure_soc_title",
        "PCA Weighted Score": "ai_exposure_score",
        "Z-Score Variance": "ai_exposure_score_variance",
    })
    .dropna(subset=["soc_code", "ai_exposure_score"])
    .copy()
)
ai_exposure_scores["soc_code"] = ai_exposure_scores["soc_code"].astype(str).str.strip()
ai_exposure_scores["ai_exposure_soc_title"] = (
    ai_exposure_scores["ai_exposure_soc_title"].astype(str).str.strip()
)
ai_exposure_scores["ai_exposure_score"] = pd.to_numeric(
    ai_exposure_scores["ai_exposure_score"],
    errors="coerce",
)
ai_exposure_scores = ai_exposure_scores.dropna(subset=["ai_exposure_score"])

# When an ACS occupation maps to multiple SOC codes, use the mean SOC score so
# each ACS worker keeps one person weight.
ai_soc_crosswalk_raw = pd.read_excel(
    "data/soc_occ_crosswalk.xlsx",
    sheet_name="NEM SOC ACS crosswalk",
    header=4,
)

ai_soc_crosswalk = (
    ai_soc_crosswalk_raw
    .rename(columns={
        "ACS code": "acs_occ_code",
        "ACS cccupational title": "acs_occupation_title",
        "National Employment Matrix code": "soc_code",
        "National Employment Matrix title": "soc_title",
    })
    .dropna(subset=["acs_occ_code", "soc_code"])
    .copy()
)
ai_soc_crosswalk["acs_occ_code"] = pd.to_numeric(
    ai_soc_crosswalk["acs_occ_code"],
    errors="coerce",
).astype("Int64")
ai_soc_crosswalk = ai_soc_crosswalk.dropna(subset=["acs_occ_code"])
ai_soc_crosswalk["soc_code"] = ai_soc_crosswalk["soc_code"].astype(str).str.strip()
ai_soc_crosswalk["soc_title"] = ai_soc_crosswalk["soc_title"].astype(str).str.strip()
ai_soc_crosswalk["acs_occupation_title"] = (
    ai_soc_crosswalk["acs_occupation_title"].astype(str).str.strip()
)

ai_soc_crosswalk = ai_soc_crosswalk.merge(
    ai_exposure_scores[["soc_code", "ai_exposure_score", "ai_exposure_soc_title"]],
    on="soc_code",
    how="left",
)

acs_ai_exposure_lookup = (
    ai_soc_crosswalk
    .groupby("acs_occ_code", as_index=True)
    .agg(
        ai_exposure_occupation_title=("acs_occupation_title", "first"),
        ai_exposure_soc_codes=("soc_code", join_unique_text),
        ai_exposure_score=("ai_exposure_score", "mean"),
        ai_exposure_score_min=("ai_exposure_score", "min"),
        ai_exposure_score_max=("ai_exposure_score", "max"),
        ai_exposure_matched_soc_codes=("ai_exposure_score", "count"),
        ai_exposure_total_soc_codes=("soc_code", "nunique"),
    )
)

ipums_df = attach_ai_exposure(ipums_df, acs_ai_exposure_lookup)
foreign_born_df = attach_ai_exposure(foreign_born_df, acs_ai_exposure_lookup)
df_likely_legal = foreign_born_df[
    foreign_born_df["likely_legal"].fillna(False).astype(bool)
].copy()
df_likely_unauthorized = foreign_born_df[
    ~foreign_born_df["likely_legal"].fillna(False).astype(bool)
].copy()

matched_acs_count = acs_ai_exposure_lookup["ai_exposure_score"].notna().sum()
print(
    f"Loaded {len(ai_exposure_scores):,} detailed SOC exposure scores from "
    f"{AI_EXPOSURE_PATH} sheet '{AI_EXPOSURE_SHEET}'."
)
print(
    f"Mapped AI exposure scores to {matched_acs_count:,} of "
    f"{len(acs_ai_exposure_lookup):,} ACS occupation codes."
)

ai_exposure_mapping_summary = pd.DataFrame({
    "metric": [
        "Detailed SOC scores loaded",
        "SOC crosswalk rows",
        "ACS occupation codes in crosswalk",
        "ACS occupation codes with exposure score",
        "ACS occupation codes without exposure score",
    ],
    "value": [
        len(ai_exposure_scores),
        len(ai_soc_crosswalk),
        len(acs_ai_exposure_lookup),
        matched_acs_count,
        len(acs_ai_exposure_lookup) - matched_acs_count,
    ],
})
show_table(ai_exposure_mapping_summary)

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
    workers_with_scores = workers[workers["ai_exposure_score"].notna()].copy()
    score_weights = (
        workers_with_scores
        .groupby("ai_exposure_score", dropna=True)[WEIGHT_COL]
        .sum()
        .sort_index()
    )
    score_distributions[label] = score_weights

    weighted_workers = workers[WEIGHT_COL].sum()
    weighted_workers_with_scores = workers_with_scores[WEIGHT_COL].sum()
    if len(score_weights) > 0 and weighted_workers_with_scores > 0:
        weighted_mean_score = np.average(
            score_weights.index.to_numpy(dtype=float),
            weights=score_weights.to_numpy(dtype=float),
        )
    else:
        weighted_mean_score = np.nan

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
        "weighted_mean_ai_exposure": weighted_mean_score,
    })

all_score_values = np.concatenate([
    score_weights.index.to_numpy(dtype=float)
    for score_weights in score_distributions.values()
    if len(score_weights) > 0
])
if len(all_score_values) == 0:
    raise ValueError("No employed workers have mapped AI exposure scores.")

x_min = np.nanmin(all_score_values)
x_max = np.nanmax(all_score_values)
x_padding = max((x_max - x_min) * 0.05, 0.10)
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
    "AI Exposure Score Distribution by Worker Status",
    fontsize=15,
    fontweight="bold",
)
ax.set_xlabel("AI exposure score", fontsize=12)
ax.set_ylabel("Weighted density", fontsize=12)
ax.grid(axis="y", alpha=0.25)
ax.legend(title="Worker status", frameon=False)

plt.tight_layout()

output_dir = Path("outputs/figures")
output_dir.mkdir(parents=True, exist_ok=True)
ai_exposure_kde_png = output_dir / "ai_exposure_kde_by_worker_status.png"
ai_exposure_kde_svg = output_dir / "ai_exposure_kde_by_worker_status.svg"
fig.savefig(ai_exposure_kde_png, dpi=300, bbox_inches="tight")
fig.savefig(ai_exposure_kde_svg, bbox_inches="tight")
plt.show()

ai_exposure_status_summary = pd.DataFrame(summary_rows)
ai_exposure_status_summary.to_csv(
    output_dir / "ai_exposure_kde_by_worker_status_summary.csv",
    index=False,
)

print(f"Saved figure to {ai_exposure_kde_png} and {ai_exposure_kde_svg}.")
show_table(ai_exposure_status_summary)
