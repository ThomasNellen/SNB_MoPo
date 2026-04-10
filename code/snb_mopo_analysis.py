"""
SNB Monetary Policy Analysis
==============================
Downloads, processes, and visualises:
  - SNB policy rate (LZ)
  - Sight-deposit remuneration: rate below threshold (ZIGBL) and above threshold (ZIG)
  - Threshold factor (FREI)
  - SARON overnight fixing
  - Policy-decision announcement and implementation dates

Data source: SNB Data Portal (data.snb.ch), cube snbgwdzid
Author: SNB_MoPo project
"""

import os
import requests
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
from matplotlib.ticker import MultipleLocator, AutoMinorLocator
from datetime import datetime

# ── paths ────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR    = os.path.join(BASE_DIR, "input", "raw")
PROC_DIR   = os.path.join(BASE_DIR, "input", "processed")
FIG_DIR    = os.path.join(BASE_DIR, "output", "figures")
for d in [RAW_DIR, PROC_DIR, FIG_DIR]:
    os.makedirs(d, exist_ok=True)


# ═══════════════════════════════════════════════════════════════════════════════
# 1.  DATA DOWNLOAD
# ═══════════════════════════════════════════════════════════════════════════════

def download_snb_cube(cube_id: str, raw_dir: str = RAW_DIR, force: bool = False) -> str:
    """Download an SNB data-portal cube as CSV and save to raw_dir."""
    path = os.path.join(raw_dir, f"{cube_id}.csv")
    if os.path.exists(path) and not force:
        print(f"  [cache] {cube_id}.csv already present – skipping download.")
        return path
    url = f"https://data.snb.ch/api/cube/{cube_id}/data/csv/en"
    print(f"  [download] {url}")
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    with open(path, "wb") as fh:
        fh.write(r.content)
    print(f"  [saved]  {path}  ({os.path.getsize(path):,} bytes)")
    return path


# ═══════════════════════════════════════════════════════════════════════════════
# 2.  PARSE snbgwdzid  →  daily panel
# ═══════════════════════════════════════════════════════════════════════════════

def parse_snbgwdzid(path: str) -> pd.DataFrame:
    """
    Parse the semicolon-delimited SNB cube CSV (long format) and pivot to wide.

    Columns returned (all floats or NaN):
        LZ      – SNB policy rate (introduced Jun 2019)
        ZIG     – sight-deposit rate above threshold  (from Jan 2015)
        ZIGBL   – sight-deposit rate below threshold  (from Sep 2022; 0 before)
        FREI    – threshold factor                    (from Nov 2019; 20 before)
        ZIABP   – discount in basis points
        SARON   – SARON overnight fixing              (from Aug 2009)
        ENG     – special rate (Sondersatz)
    """
    records = []
    with open(path, encoding="utf-8-sig") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            if line.startswith('"CubeId"') or line.startswith('"PublishingDate"') or \
               line.startswith('"Date"'):
                continue
            parts = line.split(";")
            if len(parts) < 2:
                continue
            date_str = parts[0].strip('"')
            dim      = parts[1].strip('"')
            val_str  = parts[2].strip('"') if len(parts) > 2 else ""
            val      = float(val_str) if val_str else np.nan
            records.append({"date": date_str, "dim": dim, "value": val})

    df_long = pd.DataFrame(records)
    df_long["date"] = pd.to_datetime(df_long["date"])

    df = df_long.pivot_table(
        index="date", columns="dim", values="value", aggfunc="first"
    ).reset_index()
    df.columns.name = None

    # ── back-fill gaps with known historical values ──────────────────────────
    # ZIGBL was 0 % during the negative-rate era (deposits below threshold were
    # exempt); FREI was 20 from the system's introduction until Nov 2019.
    df["ZIGBL"] = df["ZIGBL"].where(df["ZIGBL"].notna(), other=np.nan)
    # Mark period before first ZIGBL entry as 0 %
    first_zigbl = df[df["ZIGBL"].notna()]["date"].min()
    df.loc[df["date"] < first_zigbl, "ZIGBL"] = 0.0

    # FREI: 20 before first entry
    first_frei = df[df["FREI"].notna()]["date"].min()
    df.loc[df["date"] < first_frei, "FREI"] = 20.0

    # Effective policy rate: use LZ when available, else ZIG (same value pre-2019)
    df["POLICY_RATE"] = df["LZ"].combine_first(df["ZIG"])

    return df


# ═══════════════════════════════════════════════════════════════════════════════
# 3.  POLICY DECISIONS TABLE
# ═══════════════════════════════════════════════════════════════════════════════

def build_policy_decisions() -> pd.DataFrame:
    """
    Hand-coded table of SNB monetary-policy decisions since Dec 2014.

    Announcement dates: official SNB press-release / assessment dates.
    Implementation dates: first trading day on which the new rate took effect
                          (verified against the snbgwdzid data-change dates).

    Columns:
        announce_date     – date of SNB press release / monetary policy assessment
        implement_date    – date the new rate/factor first appears in data
        event_type        – 'rate_change' | 'frei_change' | 'rate_intro'
        ZIG_new           – new ZIG value (NaN if unchanged)
        LZ_new            – new LZ value  (NaN if unchanged)
        FREI_new          – new FREI value (NaN if unchanged)
        label             – short description for the chart
    """
    rows = [
        # ── tiered system introduction & negative rates ──────────────────────
        dict(announce_date="2014-12-18", implement_date="2015-01-22",
             event_type="rate_change",
             ZIG_new=-0.25, LZ_new=np.nan, FREI_new=20.0,
             label="Neg. rates announced\n(−0.25 % / factor 20)"),
        dict(announce_date="2015-01-15", implement_date="2015-01-22",
             event_type="rate_change",
             ZIG_new=-0.75, LZ_new=np.nan, FREI_new=np.nan,
             label="EUR/CHF floor drop\n& cut to −0.75 %"),
        # ── policy-rate concept formally introduced ──────────────────────────
        dict(announce_date="2019-06-13", implement_date="2019-06-13",
             event_type="rate_intro",
             ZIG_new=np.nan, LZ_new=-0.75, FREI_new=np.nan,
             label="SNB policy rate\nintroduced (−0.75 %)"),
        # ── threshold-factor changes ─────────────────────────────────────────
        dict(announce_date="2019-09-19", implement_date="2019-11-01",
             event_type="frei_change",
             ZIG_new=np.nan, LZ_new=np.nan, FREI_new=25.0,
             label="Factor 20→25"),
        dict(announce_date="2020-03-19", implement_date="2020-04-01",
             event_type="frei_change",
             ZIG_new=np.nan, LZ_new=np.nan, FREI_new=30.0,
             label="Factor 25→30\n(COVID)"),
        # ── tightening cycle 2022-2023 ───────────────────────────────────────
        dict(announce_date="2022-06-16", implement_date="2022-06-17",
             event_type="rate_change",
             ZIG_new=-0.25, LZ_new=-0.25, FREI_new=np.nan,
             label="+50 bp: −0.75→−0.25 %"),
        dict(announce_date="2022-06-16", implement_date="2022-07-01",
             event_type="frei_change",
             ZIG_new=np.nan, LZ_new=np.nan, FREI_new=28.0,
             label="Factor 30→28"),
        dict(announce_date="2022-09-22", implement_date="2022-09-23",
             event_type="rate_change",
             ZIG_new=0.00, LZ_new=0.50, FREI_new=np.nan,
             label="+75 bp: −0.25→+0.50 %"),
        dict(announce_date="2022-12-15", implement_date="2022-12-16",
             event_type="rate_change",
             ZIG_new=0.50, LZ_new=1.00, FREI_new=np.nan,
             label="+50 bp: 0.50→1.00 %"),
        dict(announce_date="2023-03-23", implement_date="2023-03-24",
             event_type="rate_change",
             ZIG_new=1.00, LZ_new=1.50, FREI_new=np.nan,
             label="+50 bp: 1.00→1.50 %"),
        dict(announce_date="2023-06-22", implement_date="2023-06-23",
             event_type="rate_change",
             ZIG_new=1.25, LZ_new=1.75, FREI_new=np.nan,
             label="+25 bp: 1.50→1.75 %"),
        dict(announce_date="2023-09-21", implement_date="2023-12-01",
             event_type="frei_change",
             ZIG_new=np.nan, LZ_new=np.nan, FREI_new=25.0,
             label="Factor 28→25"),
        # ── easing cycle 2024-2026 ───────────────────────────────────────────
        dict(announce_date="2024-03-21", implement_date="2024-03-22",
             event_type="rate_change",
             ZIG_new=1.00, LZ_new=1.50, FREI_new=np.nan,
             label="−25 bp: 1.75→1.50 %"),
        dict(announce_date="2024-06-20", implement_date="2024-06-21",
             event_type="rate_change",
             ZIG_new=0.75, LZ_new=1.25, FREI_new=np.nan,
             label="−25 bp: 1.50→1.25 %"),
        dict(announce_date="2024-09-26", implement_date="2024-09-27",
             event_type="rate_change",
             ZIG_new=0.50, LZ_new=1.00, FREI_new=np.nan,
             label="−25 bp: 1.25→1.00 %"),
        dict(announce_date="2024-09-26", implement_date="2024-10-01",
             event_type="frei_change",
             ZIG_new=np.nan, LZ_new=np.nan, FREI_new=22.0,
             label="Factor 25→22"),
        dict(announce_date="2024-12-12", implement_date="2024-12-13",
             event_type="rate_change",
             ZIG_new=0.00, LZ_new=0.50, FREI_new=np.nan,
             label="−50 bp: 1.00→0.50 %"),
        dict(announce_date="2024-12-12", implement_date="2025-02-03",
             event_type="frei_change",
             ZIG_new=np.nan, LZ_new=np.nan, FREI_new=20.0,
             label="Factor 22→20"),
        dict(announce_date="2025-03-20", implement_date="2025-03-21",
             event_type="rate_change",
             ZIG_new=0.00, LZ_new=0.25, FREI_new=np.nan,
             label="−25 bp: 0.50→0.25 %"),
        dict(announce_date="2025-03-20", implement_date="2025-06-02",
             event_type="frei_change",
             ZIG_new=np.nan, LZ_new=np.nan, FREI_new=18.0,
             label="Factor 20→18"),
        dict(announce_date="2025-06-19", implement_date="2025-06-20",
             event_type="rate_change",
             ZIG_new=-0.25, LZ_new=0.00, FREI_new=np.nan,
             label="−25 bp: 0.25→0.00 %\n(tiered re-introduced)"),
        dict(announce_date="2025-09-18", implement_date="2025-11-03",
             event_type="frei_change",
             ZIG_new=np.nan, LZ_new=np.nan, FREI_new=16.5,
             label="Factor 18→16.5"),
        dict(announce_date="2025-12-11", implement_date="2026-03-02",
             event_type="frei_change",
             ZIG_new=np.nan, LZ_new=np.nan, FREI_new=15.0,
             label="Factor 16.5→15"),
        dict(announce_date="2026-03-19", implement_date="2026-03-19",
             event_type="rate_change",
             ZIG_new=-0.25, LZ_new=0.00, FREI_new=np.nan,
             label="Rate unchanged\n(0.00 %)"),
    ]
    df = pd.DataFrame(rows)
    df["announce_date"]  = pd.to_datetime(df["announce_date"])
    df["implement_date"] = pd.to_datetime(df["implement_date"])
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# 4.  FIGURE
# ═══════════════════════════════════════════════════════════════════════════════

def make_figure(df: pd.DataFrame, decisions: pd.DataFrame,
                start: str = "2014-10-01",
                ylim_left: tuple = (-1.0, 2.0),
                save_path: str | None = None) -> plt.Figure:
    """
    State-of-the-art summary figure of SNB monetary policy since the tiered
    sight-deposit system was introduced in December 2014.

    Left y-axis  : interest rates in %  (SARON, ZIG, ZIGBL, LZ)
    Right y-axis : threshold factor (dimensionless)
    Vertical lines: announcement dates (dashed) / implementation dates (solid)
    Legend       : placed below the plot area
    Variable names follow SNB English-homepage terminology.
    """
    matplotlib.rcParams.update({
        "font.family":     "sans-serif",
        "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial"],
        "axes.spines.top":    False,
        "axes.linewidth":     0.8,
        "xtick.direction":    "out",
        "ytick.direction":    "out",
        "xtick.major.width":  0.8,
        "ytick.major.width":  0.8,
        "axes.grid":     True,
        "grid.color":    "#DDDDDD",
        "grid.linewidth": 0.5,
        "grid.alpha":    0.7,
    })

    mask = df["date"] >= start
    d    = df[mask].copy()

    # Extra bottom margin for the legend
    fig, ax1 = plt.subplots(figsize=(16, 8))
    fig.subplots_adjust(bottom=0.22)
    ax2 = ax1.twinx()

    # ── colour palette ───────────────────────────────────────────────────────
    # Policy rate: distinct charcoal — its own conceptual category
    C_POLICY = "#2C2C2C"    # near-black / charcoal – SNB policy rate
    # Above / below threshold: same steel-blue family, differ only in line style
    C_TIER   = "#1F6FB2"    # steel blue – both tiered remuneration rates
    # SARON
    C_SARON  = "#E07B00"    # vivid amber – SARON
    # RHS
    C_FREI   = "#6A0DAD"    # purple – threshold factor
    # Vertical event lines
    C_ANN    = "#C0392B"    # strong red   – announcement date
    C_IMP    = "#27AE60"    # green        – implementation date

    # ── plotting order ensures all lines are visible ──────────────────────────
    # 1. SARON first (thinnest, goes in background of rates)
    ax1.plot(d["date"], d["SARON"],
             color=C_SARON, linewidth=1.1, alpha=0.85, zorder=2,
             label="SARON fixing at the close of the trading day")

    # 2. Rate above threshold — solid, mid-weight
    ax1.plot(d["date"], d["ZIG"],
             color=C_TIER, linewidth=2.0, linestyle="-", zorder=3,
             label="Interest rate on sight deposits above threshold")

    # 3. Rate up to threshold — same colour, dashed, drawn on top so dashes
    #    remain visible even when the value equals the policy rate
    ax1.plot(d["date"], d["ZIGBL"],
             color=C_TIER, linewidth=2.0, linestyle=(0, (6, 3)), zorder=4,
             label="Interest rate on sight deposits up to threshold")

    # 4. SNB policy rate — charcoal solid, thickest; drawn last so it is never
    #    hidden but its colour is distinct from both tier lines
    lz_mask = d["LZ"].notna()
    ax1.plot(d.loc[lz_mask, "date"], d.loc[lz_mask, "LZ"],
             color=C_POLICY, linewidth=2.8, linestyle="-", zorder=5,
             label="SNB policy rate (from Jun 2019)")

    # ── threshold factor (RHS) ───────────────────────────────────────────────
    ax2.plot(d["date"], d["FREI"],
             color=C_FREI, linewidth=1.6, linestyle=":", zorder=3,
             label="Threshold factor (rhs)")

    # ── vertical event lines ─────────────────────────────────────────────────
    dec_mask   = decisions["announce_date"] >= pd.Timestamp(start)
    ann_dates  = sorted(set(decisions.loc[dec_mask, "announce_date"]))
    impl_dates = sorted(set(decisions.loc[dec_mask, "implement_date"]))

    for ad in ann_dates:
        ax1.axvline(ad,  color=C_ANN, linewidth=0.85, linestyle="--", alpha=0.55, zorder=1)
    for id_ in impl_dates:
        ax1.axvline(id_, color=C_IMP, linewidth=0.85, linestyle="-",  alpha=0.45, zorder=1)

    # ── zero reference ───────────────────────────────────────────────────────
    ax1.axhline(0, color="black", linewidth=0.6, linestyle="-", alpha=0.30, zorder=1)

    # ── axis limits & ticks ──────────────────────────────────────────────────
    ax1.set_ylim(ylim_left)
    ax1.yaxis.set_major_locator(MultipleLocator(0.25))
    ax1.yaxis.set_minor_locator(MultipleLocator(0.125))
    ax1.set_ylabel("Interest rate (%)", fontsize=11)

    frei_vals = d["FREI"].dropna()
    ax2.set_ylim(frei_vals.min() - 3, frei_vals.max() + 5)
    ax2.yaxis.set_major_locator(MultipleLocator(5))
    ax2.set_ylabel("Threshold factor", fontsize=11, color=C_FREI)
    ax2.tick_params(axis="y", colors=C_FREI)
    ax2.spines["right"].set_color(C_FREI)

    ax1.xaxis.set_major_locator(mdates.YearLocator())
    ax1.xaxis.set_minor_locator(mdates.MonthLocator(bymonth=[4, 7, 10]))
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax1.tick_params(axis="x", which="major", labelsize=10)
    ax1.set_xlabel("Date", fontsize=10, labelpad=6)

    # ── title ────────────────────────────────────────────────────────────────
    ax1.set_title(
        "SNB Monetary Policy & Overnight Repo Market (SARON)\n"
        "Tiered Sight-Deposit Remuneration System · Dec 2014 – present",
        fontsize=13, fontweight="bold", pad=12,
    )

    # ── legend below the axes ────────────────────────────────────────────────
    handles1, labels1 = ax1.get_legend_handles_labels()
    pairs1 = list(zip(handles1, labels1))

    handles2, labels2 = ax2.get_legend_handles_labels()

    ann_line  = mlines.Line2D([], [], color=C_ANN, linewidth=1.3,
                               linestyle="--", label="Announcement date")
    impl_line = mlines.Line2D([], [], color=C_IMP, linewidth=1.3,
                               linestyle="-",  label="Implementation date")

    all_handles = [h for h, _ in pairs1] + handles2 + [ann_line, impl_line]
    all_labels  = [l for _, l in pairs1] + labels2  + \
                  ["Announcement date", "Implementation date"]

    fig.legend(
        handles=all_handles, labels=all_labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.01),
        ncol=3,
        fontsize=8.5,
        framealpha=0.95,
        edgecolor="#AAAAAA",
        handlelength=2.2,
    )

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  [saved] {save_path}")

    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# 5.  MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("SNB Monetary Policy Analysis")
    print("=" * 60)

    # 1. Download raw data
    print("\n[1] Downloading raw data …")
    path = download_snb_cube("snbgwdzid", raw_dir=RAW_DIR)

    # 2. Parse into dataframe
    print("\n[2] Parsing snbgwdzid …")
    df = parse_snbgwdzid(path)
    print(f"    Rows: {len(df):,}  |  Date range: {df['date'].min().date()} – {df['date'].max().date()}")
    print(df[["date","LZ","ZIG","ZIGBL","FREI","SARON"]].tail(5).to_string(index=False))

    # Save processed data
    proc_path = os.path.join(PROC_DIR, "snb_rates_daily.parquet")
    df.to_parquet(proc_path, index=False)
    print(f"\n    Processed data saved → {proc_path}")

    # 3. Build policy-decisions table
    print("\n[3] Building policy-decisions table …")
    decisions = build_policy_decisions()
    dec_path  = os.path.join(PROC_DIR, "policy_decisions.parquet")
    decisions.to_parquet(dec_path, index=False)
    print(f"    {len(decisions)} decisions | saved → {dec_path}")

    # 4. Also save as CSV for inspection
    df.to_csv(os.path.join(PROC_DIR, "snb_rates_daily.csv"), index=False)
    decisions.to_csv(os.path.join(PROC_DIR, "policy_decisions.csv"), index=False)

    # 5. Make figure
    print("\n[4] Generating figure …")
    fig_path = os.path.join(FIG_DIR, "snb_mopo_tiered.png")
    fig = make_figure(df, decisions, save_path=fig_path)

    # Also save as PDF for publication quality
    fig.savefig(os.path.join(FIG_DIR, "snb_mopo_tiered.pdf"),
                bbox_inches="tight")
    print(f"    PDF  → {os.path.join(FIG_DIR, 'snb_mopo_tiered.pdf')}")

    plt.show()
    print("\n[Done]")


if __name__ == "__main__":
    main()
