"""Plot likely undocumented wages against average establishment size by sector.

The local QCEW extract has only ``size_code == 0`` rows, which are "All
establishment sizes." This figure therefore uses average establishment size
within each 2-digit NAICS sector: annual average employment divided by annual
average establishments.
"""

from __future__ import annotations

from pathlib import Path
import re

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
QCEW_INDUSTRY_DIR = ROOT / "data" / "2024.annual.by_industry"
OUTPUT_TABLE = (
    ROOT
    / "outputs"
    / "tables"
    / "undocumented_wages_by_avg_establishment_size_sector.csv"
)
OUTPUT_PNG = (
    ROOT
    / "outputs"
    / "figures"
    / "undocumented_wages_by_avg_establishment_size_sector.png"
)
OUTPUT_SVG = (
    ROOT
    / "outputs"
    / "figures"
    / "undocumented_wages_by_avg_establishment_size_sector.svg"
)

TWO_DIGIT_NAICS_FILE_RE = re.compile(
    r"^2024\.annual (?P<naics_code>\d{2}(?:-\d{2})?) "
    r"NAICS (?P=naics_code) .+\.csv$"
)
EXCLUDED_NAICS_CODES = {"99"}

QCEW_USECOLS = [
    "area_fips",
    "own_title",
    "industry_code",
    "industry_title",
    "size_code",
    "size_title",
    "annual_avg_estabs_count",
    "annual_avg_emplvl",
    "total_annual_wages",
]

ACS_TO_QCEW_NAICS_SECTOR = {
    "31": "31-33",
    "32": "31-33",
    "33": "31-33",
    "44": "44-45",
    "45": "44-45",
    "48": "48-49",
    "49": "48-49",
}

WAGE_COL = "INCWAGE"
FULL_TIME_MIN_HOURS = 35
MIN_WEEKS_WORKED = 50
FULL_TIME_HOURS_CANDIDATES = ["UHRSWORK", "UHRSWORKLY"]
DEFAULT_MIN_UNWEIGHTED_WORKERS = 30
DEFAULT_MIN_WEIGHTED_WORKERS = 10_000
DEFAULT_TOP_CLUSTER_SECTORS = 3
HIGHLIGHT_NAICS_CODES = {"11", "23", "56"}
NATIVE_BORN_BPL_THRESHOLD = 150


def show_table(df: pd.DataFrame) -> None:
    try:
        display(df)
    except NameError:
        print(df.to_string(index=False))


def running_in_notebook() -> bool:
    try:
        shell_name = get_ipython().__class__.__name__
    except NameError:
        return False
    return shell_name == "ZMQInteractiveShell"


def numeric_series(df: pd.DataFrame, column: str) -> pd.Series:
    series = df[column]
    if pd.api.types.is_numeric_dtype(series):
        return series
    return pd.to_numeric(series, errors="coerce")


def naics_sort_key(naics_code: str) -> tuple[int, int]:
    parts = [int(part) for part in str(naics_code).split("-")]
    if len(parts) == 1:
        parts.append(parts[0])
    return parts[0], parts[1]


def clean_industry_title(naics_code: str, industry_title: str) -> str:
    title = str(industry_title).strip()
    prefix = f"NAICS {naics_code} "
    if title.startswith(prefix):
        title = title[len(prefix):]
    return title


def find_two_digit_naics_files(
    industry_dir: Path = QCEW_INDUSTRY_DIR,
) -> list[Path]:
    if not industry_dir.exists():
        raise FileNotFoundError(f"QCEW industry folder not found: {industry_dir}")

    paths = []
    for path in industry_dir.glob("*.csv"):
        match = TWO_DIGIT_NAICS_FILE_RE.match(path.name)
        if match is None:
            continue
        if match.group("naics_code") in EXCLUDED_NAICS_CODES:
            continue
        paths.append(path)

    return sorted(
        paths,
        key=lambda p: naics_sort_key(
            TWO_DIGIT_NAICS_FILE_RE.match(p.name).group("naics_code")
        ),
    )


def read_qcew_average_establishment_size(
    industry_dir: Path = QCEW_INDUSTRY_DIR,
) -> pd.DataFrame:
    rows = []

    for path in find_two_digit_naics_files(industry_dir):
        match = TWO_DIGIT_NAICS_FILE_RE.match(path.name)
        if match is None:
            continue

        naics_code = match.group("naics_code")
        data = pd.read_csv(path, usecols=QCEW_USECOLS)
        data["size_code"] = pd.to_numeric(data["size_code"], errors="coerce")
        us_all_size_rows = data[
            data["area_fips"].astype(str).eq("US000") & data["size_code"].eq(0)
        ].copy()
        if us_all_size_rows.empty:
            print(f"Skipping {path.name}: no US000 all-size rows found.")
            continue

        for col in [
            "annual_avg_estabs_count",
            "annual_avg_emplvl",
            "total_annual_wages",
        ]:
            us_all_size_rows[col] = pd.to_numeric(
                us_all_size_rows[col],
                errors="coerce",
            ).fillna(0)

        us_all_size_rows["naics_code"] = naics_code
        us_all_size_rows["source_file"] = path.name
        rows.append(us_all_size_rows)

    if not rows:
        raise ValueError(f"No 2-digit NAICS sector files found in {industry_dir}.")

    qcew_rows = pd.concat(rows, ignore_index=True)
    qcew_sector = (
        qcew_rows
        .groupby(["naics_code", "industry_code", "industry_title", "source_file"], as_index=False)
        .agg(
            qcew_annual_avg_establishments=(
                "annual_avg_estabs_count",
                "sum",
            ),
            qcew_annual_avg_employment=("annual_avg_emplvl", "sum"),
            qcew_total_annual_wages=("total_annual_wages", "sum"),
            ownership_rows_summed=(
                "own_title",
                lambda values: "; ".join(sorted(values.astype(str).unique())),
            ),
        )
    )

    qcew_sector["avg_workers_per_establishment"] = (
        qcew_sector["qcew_annual_avg_employment"]
        / qcew_sector["qcew_annual_avg_establishments"].replace(0, pd.NA)
    )
    qcew_sector["qcew_avg_annual_pay_all_workers"] = (
        qcew_sector["qcew_total_annual_wages"]
        / qcew_sector["qcew_annual_avg_employment"].replace(0, pd.NA)
    )
    qcew_sector["industry_short_title"] = [
        clean_industry_title(code, title)
        for code, title in zip(
            qcew_sector["naics_code"],
            qcew_sector["industry_title"],
        )
    ]
    qcew_sector["industry_label"] = (
        qcew_sector["naics_code"] + " " + qcew_sector["industry_short_title"]
    )

    sort_keys = qcew_sector["naics_code"].map(naics_sort_key)
    qcew_sector["naics_sort_major"] = [key[0] for key in sort_keys]
    qcew_sector["naics_sort_minor"] = [key[1] for key in sort_keys]

    return (
        qcew_sector
        .sort_values(["naics_sort_major", "naics_sort_minor"])
        .reset_index(drop=True)
    )


def add_qcew_naics_sector(df: pd.DataFrame) -> pd.DataFrame:
    if "INDNAICS" not in df.columns:
        raise KeyError("Need INDNAICS to map workers to NAICS sectors.")

    out = df.copy()
    out["acs_naics_2_digit"] = (
        out["INDNAICS"].astype("string").str.strip().str.extract(r"^(\d{2})", expand=False)
    )
    out["naics_code"] = out["acs_naics_2_digit"].map(
        ACS_TO_QCEW_NAICS_SECTOR
    ).fillna(out["acs_naics_2_digit"])
    out.loc[out["naics_code"].isin(EXCLUDED_NAICS_CODES), "naics_code"] = pd.NA
    return out


def build_sector_cluster_counts(
    worker_df: pd.DataFrame,
    status_mask: pd.Series,
    weight_col: str,
    value_prefix: str,
) -> pd.DataFrame:
    """Count where a worker group clusters across 2-digit NAICS sectors."""
    sector_df = add_qcew_naics_sector(worker_df)
    if weight_col not in sector_df.columns:
        sector_df[weight_col] = pd.to_numeric(
            sector_df.get("PERWT", 0),
            errors="coerce",
        ).fillna(0)

    weights = numeric_series(sector_df, weight_col).fillna(0)
    valid_mask = (
        status_mask.reindex(sector_df.index, fill_value=False).astype(bool)
        & sector_df["naics_code"].notna()
        & (weights > 0)
    )
    if "EMPSTAT" in sector_df.columns:
        valid_mask &= numeric_series(sector_df, "EMPSTAT").eq(1)

    grouped = (
        sector_df.loc[valid_mask, ["naics_code", weight_col]]
        .assign(**{weight_col: weights.loc[valid_mask]})
        .groupby("naics_code", as_index=False)
        .agg(
            **{
                f"{value_prefix}_weighted_workers": (weight_col, "sum"),
                f"{value_prefix}_unweighted_workers": (weight_col, "size"),
            }
        )
        .sort_values(f"{value_prefix}_weighted_workers", ascending=False)
        .reset_index(drop=True)
    )

    grouped[f"{value_prefix}_cluster_rank"] = range(1, len(grouped) + 1)
    return grouped


def build_comparison_cluster_counts(weight_col: str) -> pd.DataFrame:
    top_n = int(globals().get("WORKER_CLUSTER_TOP_N", DEFAULT_TOP_CLUSTER_SECTORS))
    cluster_tables = []

    if "foreign_born_df" in globals() and "likely_legal" in foreign_born_df.columns:
        likely_legal_mask = foreign_born_df["likely_legal"].fillna(False).astype(bool)
        documented_counts = build_sector_cluster_counts(
            foreign_born_df,
            likely_legal_mask,
            weight_col,
            "likely_legal_immigrant",
        )
        documented_counts.loc[
            documented_counts["likely_legal_immigrant_cluster_rank"] > top_n,
            "likely_legal_immigrant_cluster_rank",
        ] = pd.NA
        documented_counts["likely_legal_immigrant_cluster_rank"] = (
            documented_counts["likely_legal_immigrant_cluster_rank"].astype("Int64")
        )
        cluster_tables.append(documented_counts)

    if "ipums_df" in globals() and "BPL" in ipums_df.columns:
        native_mask = (
            pd.to_numeric(ipums_df["BPL"], errors="coerce")
            < NATIVE_BORN_BPL_THRESHOLD
        )
        native_counts = build_sector_cluster_counts(
            ipums_df,
            native_mask,
            weight_col,
            "native_born",
        )
        native_counts.loc[
            native_counts["native_born_cluster_rank"] > top_n,
            "native_born_cluster_rank",
        ] = pd.NA
        native_counts["native_born_cluster_rank"] = native_counts[
            "native_born_cluster_rank"
        ].astype("Int64")
        cluster_tables.append(native_counts)

    if not cluster_tables:
        return pd.DataFrame({"naics_code": pd.Series(dtype="string")})

    merged = cluster_tables[0]
    for table in cluster_tables[1:]:
        merged = merged.merge(table, on="naics_code", how="outer")
    return merged


def build_undocumented_sector_wages(
    foreign_born: pd.DataFrame,
    weight_col: str,
) -> pd.DataFrame:
    if "likely_legal" not in foreign_born.columns:
        raise RuntimeError("Run the likely-legal classification cells first.")
    if WAGE_COL not in foreign_born.columns:
        raise KeyError(f"Need {WAGE_COL} in foreign_born_df.")

    worker_df = add_qcew_naics_sector(foreign_born)
    if weight_col not in worker_df.columns:
        worker_df[weight_col] = pd.to_numeric(
            worker_df.get("PERWT", 0),
            errors="coerce",
        ).fillna(0)

    likely_undocumented = ~worker_df["likely_legal"].fillna(False).astype(bool)
    wages = numeric_series(worker_df, WAGE_COL)
    weights = numeric_series(worker_df, weight_col).fillna(0)

    valid_mask = (
        likely_undocumented
        & worker_df["naics_code"].notna()
        & wages.notna()
        & (wages > 0)
        & (wages < 999_998)
        & (weights > 0)
    )
    if "EMPSTAT" in worker_df.columns:
        valid_mask &= numeric_series(worker_df, "EMPSTAT").eq(1)

    full_time_full_year_only = bool(
        globals().get("UNDOC_WAGE_FULL_TIME_FULL_YEAR_ONLY", False)
    )
    if full_time_full_year_only:
        hours_col = next(
            (col for col in FULL_TIME_HOURS_CANDIDATES if col in worker_df.columns),
            None,
        )
        if hours_col is None or "WKSWORK1" not in worker_df.columns:
            raise KeyError(
                "Full-time/full-year filtering needs UHRSWORK or UHRSWORKLY "
                "and WKSWORK1."
            )
        valid_mask &= numeric_series(worker_df, hours_col).ge(FULL_TIME_MIN_HOURS)
        valid_mask &= numeric_series(worker_df, "WKSWORK1").ge(MIN_WEEKS_WORKED)

    wage_df = worker_df.loc[valid_mask, ["naics_code", WAGE_COL, weight_col]].copy()
    wage_df[WAGE_COL] = wages.loc[wage_df.index]
    wage_df[weight_col] = weights.loc[wage_df.index]
    wage_df["weighted_wage"] = wage_df[WAGE_COL] * wage_df[weight_col]

    grouped = (
        wage_df
        .groupby("naics_code", as_index=False)
        .agg(
            undocumented_weighted_workers=(weight_col, "sum"),
            undocumented_unweighted_workers=(weight_col, "size"),
            undocumented_weighted_wage_sum=("weighted_wage", "sum"),
        )
    )
    grouped["undocumented_avg_annual_wage"] = (
        grouped["undocumented_weighted_wage_sum"]
        / grouped["undocumented_weighted_workers"]
    )

    min_unweighted_workers = int(
        globals().get(
            "MIN_UNDOC_SECTOR_WAGE_RECORDS",
            DEFAULT_MIN_UNWEIGHTED_WORKERS,
        )
    )
    min_weighted_workers = float(
        globals().get(
            "MIN_UNDOC_SECTOR_WEIGHTED_WORKERS",
            DEFAULT_MIN_WEIGHTED_WORKERS,
        )
    )
    grouped["passes_sample_floor"] = (
        grouped["undocumented_unweighted_workers"].ge(min_unweighted_workers)
        & grouped["undocumented_weighted_workers"].ge(min_weighted_workers)
    )

    return grouped


def build_analysis_table() -> pd.DataFrame:
    if "foreign_born_df" not in globals():
        raise RuntimeError("Run this from final_report.ipynb after foreign_born_df exists.")

    weight_col = globals().get("WEIGHT_COL", "PERWT_num")
    sector_wages = build_undocumented_sector_wages(foreign_born_df, weight_col)
    qcew_sector_size = read_qcew_average_establishment_size()
    comparison_cluster_counts = build_comparison_cluster_counts(weight_col)

    merged = (
        sector_wages
        .merge(qcew_sector_size, on="naics_code", how="inner")
        .merge(comparison_cluster_counts, on="naics_code", how="left")
        .sort_values("avg_workers_per_establishment")
        .reset_index(drop=True)
    )
    if merged.empty:
        raise ValueError("No undocumented wage sectors could be linked to QCEW sectors.")

    return merged


def plot_wage_by_establishment_size(
    analysis_table: pd.DataFrame,
) -> tuple[Path, Path]:
    theme_colors = globals().get("VALDOS_COLORS", {})
    valdos_blue = theme_colors.get("primary", "#272760")
    valdos_blue_dark = theme_colors.get("primary_dark", "#1E1F4B")
    point_color = "#A9B0B6"
    point_edge = "#707A82"
    undocumented_highlight_color = "#6F86C7"
    undocumented_highlight_edge = "#43599B"
    documented_cluster_color = valdos_blue
    documented_cluster_edge = valdos_blue_dark
    native_cluster_color = "#B7C4E7"
    native_cluster_edge = "#5D72B2"
    text_color = theme_colors.get("text", "#222222")
    top_n = int(globals().get("WORKER_CLUSTER_TOP_N", DEFAULT_TOP_CLUSTER_SECTORS))

    plot_df = analysis_table[analysis_table["passes_sample_floor"]].copy()
    if plot_df.empty:
        raise ValueError(
            "All linked sectors were below the sample floor; lower "
            "MIN_UNDOC_SECTOR_WAGE_RECORDS or MIN_UNDOC_SECTOR_WEIGHTED_WORKERS."
        )

    max_workers = plot_df["undocumented_weighted_workers"].max()
    plot_df["point_size"] = 70 + 420 * (
        plot_df["undocumented_weighted_workers"] / max_workers
    ).pow(0.5)
    plot_df["is_undocumented_high_share_sector"] = (
        plot_df["naics_code"].astype(str).isin(HIGHLIGHT_NAICS_CODES)
    )
    plot_df["is_likely_legal_immigrant_cluster"] = (
        plot_df.get(
            "likely_legal_immigrant_cluster_rank",
            pd.Series(pd.NA, index=plot_df.index),
        )
        .notna()
    )
    plot_df["is_native_born_cluster"] = (
        plot_df.get(
            "native_born_cluster_rank",
            pd.Series(pd.NA, index=plot_df.index),
        )
        .notna()
    )

    plot_df["has_any_highlight"] = (
        plot_df["is_undocumented_high_share_sector"]
        | plot_df["is_likely_legal_immigrant_cluster"]
        | plot_df["is_native_born_cluster"]
    )

    fig, ax = plt.subplots(figsize=(12, 7))

    ax.scatter(
        plot_df["avg_workers_per_establishment"],
        plot_df["undocumented_avg_annual_wage"],
        s=plot_df["point_size"],
        color=point_color,
        edgecolor=point_edge,
        linewidth=0.7,
        alpha=0.45,
        zorder=1,
        label="_nolegend_",
    )

    other_df = plot_df[~plot_df["has_any_highlight"]]
    if not other_df.empty:
        ax.scatter(
            other_df["avg_workers_per_establishment"],
            other_df["undocumented_avg_annual_wage"],
            s=other_df["point_size"],
            color=point_color,
            edgecolor=point_edge,
            linewidth=0.7,
            alpha=0.6,
            zorder=2,
            label="Other sectors",
        )

    undoc_df = plot_df[plot_df["is_undocumented_high_share_sector"]]
    if not undoc_df.empty:
        ax.scatter(
            undoc_df["avg_workers_per_establishment"],
            undoc_df["undocumented_avg_annual_wage"],
            s=undoc_df["point_size"],
            color=undocumented_highlight_color,
            edgecolor=undocumented_highlight_edge,
            linewidth=1.2,
            alpha=0.9,
            zorder=3,
            label="Likely undocumented high-share sectors",
        )

    documented_df = plot_df[plot_df["is_likely_legal_immigrant_cluster"]]
    if not documented_df.empty:
        ax.scatter(
            documented_df["avg_workers_per_establishment"],
            documented_df["undocumented_avg_annual_wage"],
            s=documented_df["point_size"],
            color=documented_cluster_color,
            edgecolor=documented_cluster_edge,
            linewidth=1.3,
            alpha=0.95,
            zorder=4,
            label=f"Top {top_n} likely legal immigrant sectors",
        )

    native_df = plot_df[plot_df["is_native_born_cluster"]]
    if not native_df.empty:
        ax.scatter(
            native_df["avg_workers_per_establishment"],
            native_df["undocumented_avg_annual_wage"],
            s=native_df["point_size"] * 1.14,
            facecolors="none",
            edgecolor=native_cluster_edge,
            linewidth=2.3,
            alpha=0.95,
            zorder=5,
            label=f"Top {top_n} native-born sectors",
        )

    for _, row in plot_df.iterrows():
        if row["is_native_born_cluster"]:
            annotation_color = native_cluster_edge
            annotation_weight = "bold"
            annotation_zorder = 6
        elif row["is_likely_legal_immigrant_cluster"]:
            annotation_color = documented_cluster_edge
            annotation_weight = "bold"
            annotation_zorder = 5
        elif row["is_undocumented_high_share_sector"]:
            annotation_color = undocumented_highlight_edge
            annotation_weight = "bold"
            annotation_zorder = 4
        else:
            annotation_color = text_color
            annotation_weight = "normal"
            annotation_zorder = 3

        ax.annotate(
            row["naics_code"],
            (
                row["avg_workers_per_establishment"],
                row["undocumented_avg_annual_wage"],
            ),
            xytext=(5, 4),
            textcoords="offset points",
            fontsize=8.5,
            fontweight=annotation_weight,
            color=annotation_color,
            zorder=annotation_zorder,
        )

    ax.set_title(
        "Undocumented Wages vs. Average Establishment Size by NAICS 2-Digit Sector",
        fontsize=15,
        fontweight="bold",
        color=text_color,
    )
    ax.set_xlabel(
        "Average establishment size (QCEW annual average employment per establishment)",
        fontsize=11,
        color=text_color,
    )
    ax.set_ylabel(
        "Weighted average annual wage income, likely undocumented workers",
        fontsize=11,
        color=text_color,
    )
    ax.yaxis.set_major_formatter(mtick.StrMethodFormatter("${x:,.0f}"))
    ax.xaxis.set_major_formatter(mtick.StrMethodFormatter("{x:,.0f}"))
    ax.grid(alpha=0.25)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="upper right")

    plt.tight_layout()

    OUTPUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PNG, dpi=300, bbox_inches="tight")
    fig.savefig(OUTPUT_SVG, bbox_inches="tight")

    show_figure = globals().get("UNDOC_ESTAB_SHOW_FIGURE")
    if show_figure is None:
        show_figure = running_in_notebook()

    if show_figure:
        plt.show()
    else:
        plt.close(fig)

    return OUTPUT_PNG, OUTPUT_SVG


def main() -> pd.DataFrame:
    global undocumented_wages_establishment_size

    undocumented_wages_establishment_size = build_analysis_table()

    OUTPUT_TABLE.parent.mkdir(parents=True, exist_ok=True)
    undocumented_wages_establishment_size.to_csv(OUTPUT_TABLE, index=False)
    plot_wage_by_establishment_size(undocumented_wages_establishment_size)

    display_cols = [
        "naics_code",
        "industry_short_title",
        "avg_workers_per_establishment",
        "undocumented_avg_annual_wage",
        "undocumented_weighted_workers",
        "undocumented_unweighted_workers",
        "passes_sample_floor",
    ]
    comparison_cols = [
        "likely_legal_immigrant_weighted_workers",
        "likely_legal_immigrant_cluster_rank",
        "native_born_weighted_workers",
        "native_born_cluster_rank",
    ]
    display_cols.extend(
        col
        for col in comparison_cols
        if col in undocumented_wages_establishment_size.columns
    )
    display_table = undocumented_wages_establishment_size[display_cols].assign(
        avg_workers_per_establishment=lambda df: df[
            "avg_workers_per_establishment"
        ].round(1),
        undocumented_avg_annual_wage=lambda df: df[
            "undocumented_avg_annual_wage"
        ].round(0).astype("Int64"),
        undocumented_weighted_workers=lambda df: df[
            "undocumented_weighted_workers"
        ].round(0).astype("Int64"),
    )
    for count_col in [
        "likely_legal_immigrant_weighted_workers",
        "native_born_weighted_workers",
    ]:
        if count_col in display_table.columns:
            display_table[count_col] = (
                display_table[count_col].round(0).astype("Int64")
            )
    for rank_col in [
        "likely_legal_immigrant_cluster_rank",
        "native_born_cluster_rank",
    ]:
        if rank_col in display_table.columns:
            display_table[rank_col] = display_table[rank_col].astype("Int64")

    print(
        "Saved likely undocumented wage by average establishment size table "
        f"to {OUTPUT_TABLE}."
    )
    print(f"Saved figure to {OUTPUT_PNG} and {OUTPUT_SVG}.")
    print(
        "Note: the local QCEW extract only has size_code 0 "
        "(All establishment sizes), so this uses average workers per "
        "establishment rather than establishment-size buckets."
    )
    show_table(display_table)

    return undocumented_wages_establishment_size


if __name__ == "__main__":
    main()
