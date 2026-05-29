"""Create a CA/TX/FL likely undocumented count-and-share figure.

The likely-undocumented definition follows the workflow in
acs_detailed_estimates.ipynb. The numerator uses the notebook's
foreign-born ages 18-64 restriction; the denominator is the full weighted
state population or weighted state labor force ages 18-64 from the same ACS
extract.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "acs_data.dat"
OUT_DIR = ROOT / "outputs" / "figures"
TABLE_PATH = OUT_DIR / "ca_tx_fl_likely_unauthorized_counts_share.csv"
PNG_PATH = OUT_DIR / "ca_tx_fl_likely_unauthorized_counts_share.png"
SVG_PATH = OUT_DIR / "ca_tx_fl_likely_unauthorized_counts_share.svg"

STATE_NAMES = {
    6: "California",
    48: "Texas",
    12: "Florida",
}
STATE_ORDER = [6, 48, 12]


def parse_int(value: str, default: int = 0) -> int:
    """Parse an IPUMS fixed-width numeric field."""
    value = value.strip()
    if not value:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def read_state_denominators_from_raw() -> tuple[dict[int, float], dict[int, float]]:
    """Read age-18-64 state population and age-18-64 labor-force denominators."""
    full_state_weight = {statefip: 0.0 for statefip in STATE_ORDER}
    full_state_labor_force_18_64_weight = {statefip: 0.0 for statefip in STATE_ORDER}

    with DATA_PATH.open("rt", encoding="latin1") as data_file:
        for line in data_file:
            statefip = parse_int(line[60:62])
            if statefip not in STATE_NAMES:
                continue

            age = parse_int(line[122:125])
            perwt = parse_int(line[105:115]) / 100
            if 18 <= age <= 64:
                full_state_weight[statefip] += perwt

            labforce = parse_int(line[162:163])
            if labforce == 2 and 18 <= age <= 64:
                full_state_labor_force_18_64_weight[statefip] += perwt

    return full_state_weight, full_state_labor_force_18_64_weight


def read_target_population() -> tuple[pd.DataFrame, dict[int, float], dict[int, float]]:
    """Read only the rows/fields needed for the figure.

    In the raw fixed-width extract, PERWT has two implied decimal places.
    """
    full_state_weight = {statefip: 0.0 for statefip in STATE_ORDER}
    full_state_labor_force_18_64_weight = {
        statefip: 0.0 for statefip in STATE_ORDER
    }
    cols: dict[str, list[int | float]] = {
        "YEAR": [],
        "CBSERIAL": [],
        "STATEFIP": [],
        "PERNUM": [],
        "PERWT": [],
        "SPLOC": [],
        "AGE": [],
        "BPL": [],
        "CITIZEN": [],
        "YRIMMIG": [],
        "HINSCAID": [],
        "HINSCARE": [],
        "EDUC": [],
        "LABFORCE": [],
        "CLASSWKR": [],
        "OCC": [],
        "INCSS": [],
        "INCWELFR": [],
        "VETSTAT": [],
    }

    with DATA_PATH.open("rt", encoding="latin1") as data_file:
        for line_number, line in enumerate(data_file, start=1):
            statefip = parse_int(line[60:62])
            if statefip not in STATE_NAMES:
                continue

            perwt = parse_int(line[105:115]) / 100
            labforce = parse_int(line[162:163])

            age = parse_int(line[122:125])
            if 18 <= age <= 64:
                full_state_weight[statefip] += perwt
            if labforce == 2 and 18 <= age <= 64:
                full_state_labor_force_18_64_weight[statefip] += perwt

            bpl = parse_int(line[129:132])
            if not (18 <= age <= 64 and bpl >= 150):
                continue

            cols["YEAR"].append(parse_int(line[0:4]))
            cols["CBSERIAL"].append(parse_int(line[22:35]))
            cols["STATEFIP"].append(statefip)
            cols["PERNUM"].append(parse_int(line[101:105]))
            cols["PERWT"].append(perwt)
            cols["SPLOC"].append(parse_int(line[117:119]))
            cols["AGE"].append(age)
            cols["BPL"].append(bpl)
            cols["CITIZEN"].append(parse_int(line[137:138]))
            cols["YRIMMIG"].append(parse_int(line[142:146]))
            cols["HINSCAID"].append(parse_int(line[149:150]))
            cols["HINSCARE"].append(parse_int(line[150:151]))
            cols["EDUC"].append(parse_int(line[151:153]))
            cols["LABFORCE"].append(labforce)
            cols["CLASSWKR"].append(parse_int(line[163:164]))
            cols["OCC"].append(parse_int(line[166:170]))
            cols["INCSS"].append(parse_int(line[196:201]))
            cols["INCWELFR"].append(parse_int(line[201:206]))
            cols["VETSTAT"].append(parse_int(line[221:222]))

            if line_number % 1_000_000 == 0:
                print(f"Read {line_number:,} ACS records...")

    return pd.DataFrame(cols), full_state_weight, full_state_labor_force_18_64_weight


def apply_likely_legal_rules(foreign_born_df: pd.DataFrame) -> pd.Series:
    """Replicate the likely-legal rules used in the ACS notebook."""
    likely_legal = foreign_born_df["YRIMMIG"] <= 1980

    likely_legal = (
        likely_legal
        | (foreign_born_df["INCSS"] > 0)
        | (foreign_born_df["INCWELFR"] > 0)
        | (foreign_born_df["VETSTAT"] == 2)
        | (foreign_born_df["HINSCAID"] == 2)
        | (foreign_born_df["HINSCARE"] == 2)
    )

    citizen_reported = foreign_born_df["CITIZEN"] <= 2
    years_in_us = foreign_born_df["YEAR"] - foreign_born_df["YRIMMIG"]
    spouse_citizen_map = {
        (cbserial, pernum): is_citizen
        for cbserial, pernum, is_citizen in zip(
            foreign_born_df["CBSERIAL"],
            foreign_born_df["PERNUM"],
            citizen_reported,
        )
    }
    spouse_is_citizen = foreign_born_df.apply(
        lambda row: False
        if row["SPLOC"] == 0
        else spouse_citizen_map.get((row["CBSERIAL"], row["SPLOC"]), False),
        axis=1,
    )
    naturalization_timing_rule = citizen_reported & (
        (years_in_us >= 5) | ((years_in_us >= 3) & spouse_is_citizen)
    )
    likely_legal = likely_legal | naturalization_timing_rule

    likely_legal = likely_legal | foreign_born_df["CLASSWKR"].isin([25, 26, 27, 28])

    occ = foreign_born_df["OCC"]
    in_licensed_occ_range = (
        ((occ >= 2200) & (occ <= 3540))
        | ((occ >= 3700) & (occ <= 3950))
        | ((occ >= 2100) & (occ <= 2160))
        | ((occ >= 2300) & (occ <= 2550))
        | ((occ >= 9030) & (occ <= 9040))
        | ((occ >= 1300) & (occ <= 1560))
    )
    exempt_statefip = {6, 8, 10, 17, 32, 34, 50}
    licensed_rule_applies = ~foreign_born_df["STATEFIP"].isin(exempt_statefip)
    likely_legal = likely_legal | (in_licensed_occ_range & licensed_rule_applies)

    h1b_likely_legal = (foreign_born_df["EDUC"] >= 10) & (
        foreign_born_df["YRIMMIG"] >= 2020
    )
    likely_legal = likely_legal | h1b_likely_legal

    legal_map = {
        (cbserial, pernum): legal_status
        for cbserial, pernum, legal_status in zip(
            foreign_born_df["CBSERIAL"],
            foreign_born_df["PERNUM"],
            likely_legal,
        )
    }
    spouse_is_legal = foreign_born_df.apply(
        lambda row: False
        if row["SPLOC"] == 0
        else legal_map.get((row["CBSERIAL"], row["SPLOC"]), False),
        axis=1,
    )
    return likely_legal | spouse_is_legal


def build_summary(
    foreign_born_df: pd.DataFrame,
    full_state_weight: dict[int, float],
    full_state_labor_force_18_64_weight: dict[int, float],
    likely_legal_override: pd.Series | None = None,
) -> pd.DataFrame:
    likely_legal = (
        likely_legal_override
        if likely_legal_override is not None
        else apply_likely_legal_rules(foreign_born_df)
    )
    likely_unauthorized = foreign_born_df.loc[~likely_legal].copy()

    unauthorized_by_state = likely_unauthorized.groupby("STATEFIP")["PERWT"].sum()
    unauthorized_labor_force_by_state = (
        likely_unauthorized.loc[likely_unauthorized["LABFORCE"] == 2]
        .groupby("STATEFIP")["PERWT"]
        .sum()
    )
    rows = []
    for statefip in STATE_ORDER:
        count = float(unauthorized_by_state.get(statefip, 0))
        population = full_state_weight[statefip]
        labor_force_count = float(unauthorized_labor_force_by_state.get(statefip, 0))
        labor_force_population = full_state_labor_force_18_64_weight[statefip]
        rows.append(
            {
                "state": STATE_NAMES[statefip],
                "statefip": statefip,
                "likely_unauthorized_weighted_count": round(count),
                "state_population_weighted": round(population),
                "share_of_state_population": count / population,
                "share_of_state_population_pct": 100 * count / population,
                "likely_unauthorized_labor_force_weighted_count": round(
                    labor_force_count
                ),
                "state_labor_force_18_64_weighted": round(labor_force_population),
                "share_of_state_labor_force_18_64": labor_force_count
                / labor_force_population,
                "share_of_state_labor_force_18_64_pct": 100
                * labor_force_count
                / labor_force_population,
            }
        )
    return pd.DataFrame(rows)


def add_bar_labels(ax: plt.Axes, values: pd.Series, formatter) -> None:
    x_max = max(values) if len(values) else 0
    for index, value in enumerate(values):
        ax.text(
            value + x_max * 0.02,
            index,
            formatter(value),
            va="center",
            ha="left",
            fontsize=10,
            color="#222222",
        )


def create_figure(summary: pd.DataFrame) -> None:
    states = summary["state"]
    counts_millions = summary["likely_unauthorized_weighted_count"] / 1_000_000
    share_pct = summary["share_of_state_population_pct"]
    labor_force_share_pct = summary["share_of_state_labor_force_18_64_pct"]

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    fig, axes = plt.subplots(
        ncols=3,
        figsize=(14, 5.8),
        gridspec_kw={"width_ratios": [1.15, 1, 1]},
        constrained_layout=True,
    )
    fig.patch.set_facecolor("white")

    _vc = globals().get("VALDOS_COLORS")
    count_color = _vc["primary"] if _vc else "#2f6f8f"
    share_color = _vc["secondary"] if _vc else "#b5652a"
    labor_force_color = _vc["primary_dark"] if _vc else "#4b7f52"

    axes[0].barh(states, counts_millions, color=count_color)
    axes[0].invert_yaxis()
    axes[0].set_xlabel("Weighted count, millions")
    axes[0].set_title("Likely undocumented count", loc="left", fontsize=12)
    axes[0].grid(axis="x", alpha=0.25)
    add_bar_labels(axes[0], counts_millions, lambda value: f"{value:.2f}M")
    axes[0].set_xlim(0, max(counts_millions) * 1.22)

    axes[1].barh(states, share_pct, color=share_color)
    axes[1].invert_yaxis()
    axes[1].set_xlabel("Share of state population")
    axes[1].set_title("Share of state population", loc="left", fontsize=12)
    axes[1].grid(axis="x", alpha=0.25)
    add_bar_labels(axes[1], share_pct, lambda value: f"{value:.1f}%")
    axes[1].set_xlim(0, max(share_pct) * 1.22)

    axes[2].barh(states, labor_force_share_pct, color=labor_force_color)
    axes[2].invert_yaxis()
    axes[2].set_xlabel("Share of state labor force")
    axes[2].set_title("Share of labor force", loc="left", fontsize=12)
    axes[2].grid(axis="x", alpha=0.25)
    add_bar_labels(axes[2], labor_force_share_pct, lambda value: f"{value:.1f}%")
    axes[2].set_xlim(0, max(labor_force_share_pct) * 1.22)

    fig.suptitle(
        "Likely Undocumented Immigrants in California, Texas, and Florida",
        x=0.01,
        ha="left",
        fontsize=16,
        fontweight="bold",
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(PNG_PATH, dpi=300, bbox_inches="tight")
    fig.savefig(SVG_PATH, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    using_notebook_frames = (
        "ipums_df" in globals()
        and isinstance(globals().get("ipums_df"), pd.DataFrame)
        and "foreign_born_df" in globals()
        and isinstance(globals().get("foreign_born_df"), pd.DataFrame)
    )

    if using_notebook_frames:
        ipums_df_local = globals()["ipums_df"].copy()
        foreign_born_df = globals()["foreign_born_df"].copy()

        # Match the notebook's weight definition (prefer WEIGHT_COL if present).
        weight_col = globals().get("WEIGHT_COL", "PERWT_num")
        if weight_col not in ipums_df_local.columns:
            ipums_df_local[weight_col] = pd.to_numeric(
                ipums_df_local.get("PERWT", 0), errors="coerce"
            ).fillna(0)
        if weight_col not in foreign_born_df.columns:
            foreign_born_df[weight_col] = pd.to_numeric(
                foreign_born_df.get("PERWT", 0), errors="coerce"
            ).fillna(0)

        age_numeric = pd.to_numeric(ipums_df_local.get("AGE", pd.Series(index=ipums_df_local.index)), errors="coerce")
        labforce_numeric = pd.to_numeric(ipums_df_local.get("LABFORCE", 0), errors="coerce")

        full_state_weight = {
            statefip: float(
                ipums_df_local.loc[
                    (ipums_df_local["STATEFIP"] == statefip)
                    & age_numeric.between(18, 64, inclusive="both"),
                    weight_col,
                ].sum()
            )
            for statefip in STATE_ORDER
        }

        full_state_labor_force_18_64_weight = {
            statefip: float(
                ipums_df_local.loc[
                    (ipums_df_local["STATEFIP"] == statefip)
                    & (labforce_numeric == 2)
                    & age_numeric.between(18, 64, inclusive="both"),
                    weight_col,
                ].sum()
            )
            for statefip in STATE_ORDER
        }

        likely_legal_override = None
        if "likely_legal" in foreign_born_df.columns:
            likely_legal_override = foreign_born_df["likely_legal"].fillna(False).astype(bool)

        print("Using notebook DataFrames (ipums_df / foreign_born_df) for figure inputs.")
        print(f"Target foreign-born age 18-64 records: {len(foreign_born_df):,}")
        summary = build_summary(
            foreign_born_df,
            full_state_weight,
            full_state_labor_force_18_64_weight,
            likely_legal_override=likely_legal_override,
        )
    else:
        (
            foreign_born_df,
            full_state_weight,
            full_state_labor_force_18_64_weight,
        ) = read_target_population()
        print("Using standalone fixed-width ACS parser for figure inputs.")
        print(f"Target foreign-born age 18-64 records: {len(foreign_born_df):,}")

        summary = build_summary(
            foreign_born_df,
            full_state_weight,
            full_state_labor_force_18_64_weight,
        )
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary.to_csv(TABLE_PATH, index=False)
    create_figure(summary)

    print(summary.to_string(index=False))
    print(f"Wrote {PNG_PATH}")
    print(f"Wrote {SVG_PATH}")
    print(f"Wrote {TABLE_PATH}")


if __name__ == "__main__":
    main()
