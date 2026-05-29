"""Plot 2024 U.S. annual average employment by 2-digit NAICS sector.

The source folder contains many detailed QCEW annual-by-industry CSVs. This
script only reads national NAICS sector files, sums all U.S. ownership rows,
and excludes NAICS 99 Unclassified.
"""

from __future__ import annotations

from pathlib import Path
import re
import textwrap

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
QCEW_INDUSTRY_DIR = ROOT / "data" / "2024.annual.by_industry"
OUTPUT_TABLE = ROOT / "outputs" / "tables" / "qcew_2024_us_annual_avg_employment_by_naics_sector.csv"
OUTPUT_PNG = ROOT / "outputs" / "figures" / "qcew_2024_us_annual_avg_employment_by_naics_sector.png"
OUTPUT_SVG = ROOT / "outputs" / "figures" / "qcew_2024_us_annual_avg_employment_by_naics_sector.svg"

TWO_DIGIT_NAICS_FILE_RE = re.compile(
    r"^2024\.annual (?P<naics_code>\d{2}(?:-\d{2})?) "
    r"NAICS (?P=naics_code) .+\.csv$"
)
EXCLUDED_NAICS_CODES = {"99"}

QCEW_USECOLS = [
    "area_fips",
    "own_code",
    "own_title",
    "industry_code",
    "industry_title",
    "annual_avg_emplvl",
]


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


def snake_case_ownership_title(own_title: str) -> str:
    return (
        str(own_title)
        .strip()
        .lower()
        .replace(" ", "_")
        .replace("-", "_")
    )


def find_two_digit_naics_files(industry_dir: Path = QCEW_INDUSTRY_DIR) -> list[Path]:
    if not industry_dir.exists():
        raise FileNotFoundError(f"QCEW industry folder not found: {industry_dir}")

    paths = []
    for path in industry_dir.glob("*.csv"):
        match = TWO_DIGIT_NAICS_FILE_RE.match(path.name)
        if not match:
            continue
        if match.group("naics_code") in EXCLUDED_NAICS_CODES:
            continue
        paths.append(path)

    return sorted(paths, key=lambda p: naics_sort_key(TWO_DIGIT_NAICS_FILE_RE.match(p.name).group("naics_code")))


def read_us_sector_employment(industry_dir: Path = QCEW_INDUSTRY_DIR) -> pd.DataFrame:
    rows = []

    for path in find_two_digit_naics_files(industry_dir):
        match = TWO_DIGIT_NAICS_FILE_RE.match(path.name)
        if match is None:
            continue

        naics_code = match.group("naics_code")
        data = pd.read_csv(path, usecols=QCEW_USECOLS)
        us_rows = data[data["area_fips"].astype(str).eq("US000")].copy()
        if us_rows.empty:
            print(f"Skipping {path.name}: no US000 rows found.")
            continue

        us_rows["naics_code"] = naics_code
        us_rows["source_file"] = path.name
        us_rows["annual_avg_emplvl"] = pd.to_numeric(
            us_rows["annual_avg_emplvl"],
            errors="coerce",
        ).fillna(0)
        rows.append(us_rows)

    if not rows:
        raise ValueError(f"No two-digit NAICS sector files found in {industry_dir}.")

    national_ownership_rows = pd.concat(rows, ignore_index=True)

    sector_totals = (
        national_ownership_rows
        .groupby(["naics_code", "industry_code", "industry_title", "source_file"], as_index=False)
        .agg(
            annual_avg_employment=("annual_avg_emplvl", "sum"),
            ownership_rows_summed=("own_title", lambda values: "; ".join(sorted(values.astype(str).unique()))),
        )
    )

    ownership_breakout = (
        national_ownership_rows
        .assign(ownership_col=lambda df: df["own_title"].map(snake_case_ownership_title) + "_employment")
        .pivot_table(
            index=["naics_code", "industry_code", "industry_title", "source_file"],
            columns="ownership_col",
            values="annual_avg_emplvl",
            aggfunc="sum",
            fill_value=0,
        )
        .reset_index()
    )
    ownership_breakout.columns.name = None

    out = sector_totals.merge(
        ownership_breakout,
        on=["naics_code", "industry_code", "industry_title", "source_file"],
        how="left",
    )

    sort_keys = out["naics_code"].map(naics_sort_key)
    out["naics_sort_major"] = [key[0] for key in sort_keys]
    out["naics_sort_minor"] = [key[1] for key in sort_keys]
    out["industry_short_title"] = [
        clean_industry_title(code, title)
        for code, title in zip(out["naics_code"], out["industry_title"])
    ]
    out["industry_label"] = out["naics_code"] + " " + out["industry_short_title"]
    out["annual_avg_employment_millions"] = out["annual_avg_employment"] / 1_000_000

    out = (
        out
        .sort_values(["naics_sort_major", "naics_sort_minor", "naics_code"])
        .reset_index(drop=True)
    )

    return out


def plot_sector_employment(sector_employment: pd.DataFrame) -> tuple[Path, Path]:
    theme_colors = globals().get("VALDOS_COLORS", {})
    bar_color = theme_colors.get("primary", "#087E8B")
    text_color = theme_colors.get("text", "#222222")

    plot_df = sector_employment.copy()
    labels = [
        "\n".join(textwrap.wrap(label, width=42))
        for label in plot_df["industry_label"]
    ]

    fig_height = max(8, 0.42 * len(plot_df) + 2)
    fig, ax = plt.subplots(figsize=(12, fig_height))

    bars = ax.barh(labels, plot_df["annual_avg_employment_millions"], color=bar_color)
    ax.bar_label(
        bars,
        labels=[f"{value:.1f}M" for value in plot_df["annual_avg_employment_millions"]],
        padding=4,
        fontsize=9,
    )

    ax.invert_yaxis()
    ax.set_title(
        "2024 U.S. Annual Average Employment by NAICS Sector",
        fontsize=15,
        fontweight="bold",
        color=text_color,
    )
    ax.set_xlabel("Annual average employment, millions", fontsize=12, color=text_color)
    ax.set_ylabel("")
    ax.grid(axis="x", alpha=0.25)
    ax.set_axisbelow(True)

    max_value = plot_df["annual_avg_employment_millions"].max()
    if pd.notna(max_value):
        ax.set_xlim(0, max_value * 1.12)

    plt.tight_layout()

    OUTPUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PNG, dpi=300, bbox_inches="tight")
    fig.savefig(OUTPUT_SVG, bbox_inches="tight")
    show_figure = globals().get("QCEW_SHOW_FIGURE")
    if show_figure is None:
        show_figure = running_in_notebook()

    if show_figure:
        plt.show()
    else:
        plt.close(fig)

    return OUTPUT_PNG, OUTPUT_SVG


def main() -> pd.DataFrame:
    global qcew_us_sector_employment

    qcew_us_sector_employment = read_us_sector_employment()

    OUTPUT_TABLE.parent.mkdir(parents=True, exist_ok=True)
    qcew_us_sector_employment.to_csv(OUTPUT_TABLE, index=False)
    plot_sector_employment(qcew_us_sector_employment)

    print(
        f"Loaded {len(qcew_us_sector_employment):,} two-digit NAICS sectors "
        f"from {QCEW_INDUSTRY_DIR}."
    )
    print(f"Saved table to {OUTPUT_TABLE}.")
    print(f"Saved figure to {OUTPUT_PNG} and {OUTPUT_SVG}.")

    show_table(
        qcew_us_sector_employment[
            [
                "naics_code",
                "industry_short_title",
                "annual_avg_employment",
                "annual_avg_employment_millions",
                "ownership_rows_summed",
            ]
        ].assign(
            annual_avg_employment=lambda df: df["annual_avg_employment"].round(0).astype(int),
            annual_avg_employment_millions=lambda df: df["annual_avg_employment_millions"].round(2),
        )
    )

    return qcew_us_sector_employment


if __name__ == "__main__":
    main()
