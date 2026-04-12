"""
Settlement Flow Analysis
========================
Computes, for every auction in the three transaction-level datasets,
two settlement legs:

  Opening leg (+): auction settlement date, positive amount
    → cash flows FROM banking system TO SNB / EFV (liquidity absorbed)

  Closing leg (−): maturity / redemption date, negative amount
    → cash flows FROM SNB / EFV TO banking system (liquidity released)

Instruments covered and data availability:

  GMBF (EFV)             2012-01-05 → 2029-03-21  complete (both legs)
  Confederation bonds     2011-01-05 → 2064-06-25  opening legs from Jan 2011;
                                                    closing legs of pre-2011
                                                    taps are ABSENT (see note)
  SNB Bills               2022-08-15 → 2027-02-01  complete (both legs)
  SNB reverse repos       NOT AVAILABLE at transaction level —
                          only monthly balance-sheet outstanding (VRGSF)

Outputs
-------
  input/processed/settlement_flows_all.csv   — one row per (auction × leg)
  input/processed/settlement_daily.csv       — daily aggregates

Net convention: positive = net liquidity absorption (more absorbed than released)

Author: SNB_MoPo project
"""

import os
import pandas as pd
import numpy as np

# ── paths ─────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC_DIR = os.path.join(BASE_DIR, "input", "processed")

BILLS_CSV       = os.path.join(PROC_DIR, "snb_bills_auctions.csv")
GMBF_CSV        = os.path.join(PROC_DIR, "efv_gmbf_auctions.csv")
BONDS_CSV       = os.path.join(PROC_DIR, "efv_bond_auctions.csv")

FLOWS_OUT       = os.path.join(PROC_DIR, "settlement_flows_all.csv")
DAILY_OUT       = os.path.join(PROC_DIR, "settlement_daily.csv")


# ─────────────────────────────────────────────────────────────────────────────
# 1.  LOAD AND HARMONISE EACH DATASET
# ─────────────────────────────────────────────────────────────────────────────

def load_snb_bills() -> pd.DataFrame:
    """
    SNB Bills: weekly Thursday auctions, T+2 bank working days settlement.
    Amount = allotment_chf_mn (market portion; excludes issuer retention).
    Rows with missing allotment (most-recent auctions not yet published) dropped.
    """
    df = pd.read_csv(BILLS_CSV, parse_dates=["auction_date", "payment_date", "redemption_date"])
    df = df.dropna(subset=["allotment_chf_mn"])
    df = df[df["allotment_chf_mn"] > 0]
    return pd.DataFrame({
        "instrument":       "SNB Bills",
        "auction_date":     df["auction_date"],
        "settlement_date":  df["payment_date"],
        "maturity_date":    df["redemption_date"],
        "term_bucket":      df["term_bucket"],
        "isin":             df["isin"],
        "amount_chf_mn":    df["allotment_chf_mn"],
    })


def load_gmbf() -> pd.DataFrame:
    """
    GMBF: weekly Tuesday auctions, T+2 bank working days settlement.
    Amount = issue_volume_chf_mn.
    Rows with missing volume dropped (none expected given data structure).
    """
    df = pd.read_csv(GMBF_CSV, parse_dates=["auction_date", "settlement_date", "maturity_date"])
    df = df.dropna(subset=["issue_volume_chf_mn"])
    df = df[df["issue_volume_chf_mn"] > 0]
    return pd.DataFrame({
        "instrument":       "GMBF",
        "auction_date":     df["auction_date"],
        "settlement_date":  df["settlement_date"],
        "maturity_date":    df["maturity_date"],
        "term_bucket":      df["term_bucket"],
        "isin":             df["isin"],
        "amount_chf_mn":    df["issue_volume_chf_mn"],
    })


def load_bonds() -> pd.DataFrame:
    """
    Confederation bonds: monthly auctions (~T+14 settlement).
    Amount = issue_volume_chf_mn (market portion only; own-holdings excluded).

    Coverage note: our data begins with the November 2010 auction.
    Earlier tap issuances of bonds that are still outstanding (e.g. the
    1999 bond maturing 2049) are not captured → their opening legs are absent.
    The dataset is structurally complete only for bonds whose FIRST settlement
    in our data post-dates their actual first issuance, which applies cleanly
    to series started from 2011 onward.
    """
    df = pd.read_csv(BONDS_CSV, parse_dates=["auction_date", "settlement_date", "maturity_date"])
    df = df.dropna(subset=["issue_volume_chf_mn"])
    df = df[df["issue_volume_chf_mn"] > 0]
    return pd.DataFrame({
        "instrument":       "Conf. Bonds",
        "auction_date":     df["auction_date"],
        "settlement_date":  df["settlement_date"],
        "maturity_date":    df["maturity_date"],
        "term_bucket":      df["residual_maturity_yrs"].apply(
                                lambda y: f"{int(y)}yr" if pd.notna(y) else np.nan),
        "isin":             df["fungible_isin"].fillna(df["prov_isin"]),
        "amount_chf_mn":    df["issue_volume_chf_mn"],
    })


# ─────────────────────────────────────────────────────────────────────────────
# 2.  BUILD INDIVIDUAL SETTLEMENT LEGS
# ─────────────────────────────────────────────────────────────────────────────

def make_legs(df_base: pd.DataFrame) -> pd.DataFrame:
    """
    For each auction row, produce:
      - an opening row  (leg='opening', date=settlement_date, sign= +1)
      - a closing row   (leg='closing', date=maturity_date,   sign= −1)

    Signed amount convention:
      + opening  → liquidity absorbed from banking system
      − closing  → liquidity returned to banking system
    """
    shared = ["instrument", "auction_date", "term_bucket", "isin", "amount_chf_mn"]

    opening = df_base[shared + ["settlement_date"]].copy()
    opening["leg"]            = "opening"
    opening["settlement_date_leg"] = opening["settlement_date"]
    opening["signed_chf_mn"]  = +opening["amount_chf_mn"]

    closing = df_base[shared + ["maturity_date"]].copy()
    closing["leg"]            = "closing"
    closing["settlement_date_leg"] = closing["maturity_date"]
    closing["signed_chf_mn"]  = -closing["amount_chf_mn"]

    legs = pd.concat(
        [opening[["instrument","auction_date","settlement_date_leg",
                   "leg","term_bucket","isin","amount_chf_mn","signed_chf_mn"]],
         closing[["instrument","auction_date","settlement_date_leg",
                   "leg","term_bucket","isin","amount_chf_mn","signed_chf_mn"]]],
        ignore_index=True,
    )
    legs = legs.rename(columns={"settlement_date_leg": "settlement_date"})
    return legs.sort_values(["settlement_date", "instrument", "leg"]).reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# 3.  DAILY AGGREGATION
# ─────────────────────────────────────────────────────────────────────────────

def daily_aggregate(df_legs: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate all settlement legs to a daily frequency.

    Columns in output:
      date                   — calendar date
      opening_chf_mn         — total new-issue settlements (liquidity absorbed)
      closing_chf_mn         — total maturities/redemptions (positive; liquidity released)
      net_chf_mn             — opening − closing  (+ = net absorption)
      n_opening              — number of opening legs on this date
      n_closing              — number of closing legs on this date

    Per-instrument sub-totals also included (bills_, gmbf_, bonds_ prefixes).
    """
    df = df_legs.copy()
    df["date"] = pd.to_datetime(df["settlement_date"]).dt.normalize()

    # ── aggregate grand totals ─────────────────────────────────────────────
    open_ = df[df["leg"] == "opening"].groupby("date").agg(
        opening_chf_mn=("amount_chf_mn", "sum"),
        n_opening=("amount_chf_mn", "count"),
    )
    close_ = df[df["leg"] == "closing"].groupby("date").agg(
        closing_chf_mn=("amount_chf_mn", "sum"),
        n_closing=("amount_chf_mn", "count"),
    )

    daily = open_.join(close_, how="outer").fillna(0)
    daily["net_chf_mn"] = daily["opening_chf_mn"] - daily["closing_chf_mn"]
    daily["n_opening"]  = daily["n_opening"].astype(int)
    daily["n_closing"]  = daily["n_closing"].astype(int)

    # ── per-instrument sub-totals ──────────────────────────────────────────
    for inst, prefix in [
        ("SNB Bills",   "bills"),
        ("GMBF",        "gmbf"),
        ("Conf. Bonds", "bonds"),
    ]:
        sub = df[df["instrument"] == inst]
        sub_o = sub[sub["leg"] == "opening"].groupby("date")["amount_chf_mn"].sum().rename(f"{prefix}_opening_chf_mn")
        sub_c = sub[sub["leg"] == "closing"].groupby("date")["amount_chf_mn"].sum().rename(f"{prefix}_closing_chf_mn")
        daily = daily.join(sub_o, how="left").join(sub_c, how="left")
        daily[f"{prefix}_net_chf_mn"] = (
            daily[f"{prefix}_opening_chf_mn"].fillna(0)
            - daily[f"{prefix}_closing_chf_mn"].fillna(0)
        )

    daily = daily.reset_index().rename(columns={"index": "date"})
    daily = daily.sort_values("date").reset_index(drop=True)
    return daily


# ─────────────────────────────────────────────────────────────────────────────
# 4.  COVERAGE DIAGNOSTICS
# ─────────────────────────────────────────────────────────────────────────────

def coverage_report(df_flows: pd.DataFrame, df_daily: pd.DataFrame) -> None:
    """
    Print a detailed sample-period and coverage assessment.
    """
    print("\n" + "─" * 60)
    print("COVERAGE ASSESSMENT")
    print("─" * 60)

    for inst in ["SNB Bills", "GMBF", "Conf. Bonds"]:
        sub = df_flows[df_flows["instrument"] == inst]
        op  = sub[sub["leg"] == "opening"]
        cl  = sub[sub["leg"] == "closing"]
        print(f"\n{inst}:")
        print(f"  Opening legs : {op['settlement_date'].min().date()} → "
              f"{op['settlement_date'].max().date()}  ({len(op)} rows)")
        print(f"  Closing legs : {cl['settlement_date'].min().date()} → "
              f"{cl['settlement_date'].max().date()}  ({len(cl)} rows)")
        print(f"  Total notional opened : CHF {op['amount_chf_mn'].sum():>10,.0f} mn")
        print(f"  Total notional closed : CHF {cl['amount_chf_mn'].sum():>10,.0f} mn")

    # ── bond pre-2011 gap ─────────────────────────────────────────────────
    bonds_op = df_flows[(df_flows["instrument"] == "Conf. Bonds") & (df_flows["leg"] == "opening")]
    bonds_cl = df_flows[(df_flows["instrument"] == "Conf. Bonds") & (df_flows["leg"] == "closing")]

    # closing legs in window whose opening legs fall BEFORE data start
    missing_open = bonds_cl[bonds_cl["auction_date"] < bonds_op["auction_date"].min()]
    print(f"\n  Bonds — missing opening legs (tapped before Nov 2010):")
    print(f"    Closing legs with pre-2011 auction_date : {len(missing_open)}")
    print(f"    (These are tap issuances not in our auction dataset)")

    # ── daily coverage summary ────────────────────────────────────────────
    print(f"\nDaily settlement calendar:")
    print(f"  Full span  : {df_daily['date'].min().date()} → "
          f"{df_daily['date'].max().date()}")

    # Days with BOTH instruments active (GMBF + Bonds)
    overlap_start = max(
        df_flows[df_flows["instrument"] == "GMBF"]["settlement_date"].min(),
        df_flows[df_flows["instrument"] == "Conf. Bonds"]["settlement_date"].min(),
    )
    print(f"  GMBF + Bonds overlap from : {overlap_start.date()}")

    # Days with all three instruments active
    all3_start = df_flows[df_flows["instrument"] == "SNB Bills"]["settlement_date"].min()
    print(f"  All 3 instruments from    : {all3_start.date()}")

    # ── can we go back to 2005? ───────────────────────────────────────────
    print("\n" + "─" * 60)
    print("CAN WE RELIABLY COVER 2005 ONWARDS?")
    print("─" * 60)
    print("""
  Instrument          First opening leg   First closing leg
  ─────────────────   ─────────────────   ─────────────────""")
    for inst, op_start, cl_start in [
        ("SNB Bills",   "2022-08-15",  "2022-08-22"),
        ("GMBF",        "2012-01-05",  "2012-03-21"),
        ("Conf. Bonds", "2011-01-05",  "2015-06-10"),
        ("SNB Repos",   "n/a (monthly balance sheet only)", "n/a"),
    ]:
        print(f"  {inst:<20s}  {op_start:<19s}  {cl_start}")

    print("""
  Conclusion:
  ───────────
  No. The datasets do not support a reliable settlement calendar
  back to 2005. Specific gaps:

  1. GMBF data begins January 2012. Before that, no transaction-level
     data on Confederation money-market issuance exists in the EFV XLSX.

  2. Confederation bond data begins November 2010. Importantly, several
     bonds currently outstanding were first issued before this date
     (e.g. Eidg. 06.01.99/49, Eidg. 08.04.98/28, Eidg. 08.04.03/33).
     Their pre-2011 tap issuances are absent from our opening legs,
     though their eventual maturities (2026–2064) will correctly appear
     as closing legs.  This creates a structural undercounting of opening-
     leg volumes for the years 2011–2025.

  3. SNB Bills only exist from August 2022 (reactivated after positive
     rates returned). In the first episode (2008–2012) no transaction-
     level data is publicly available.

  4. SNB reverse repos: no public transaction-level database at all.
     Only a monthly outstanding (VRGSF from snbbipo) is available.

  Recommended clean sample windows:
  ──────────────────────────────────
  A. GMBF only (both legs fully observed): Jan 2012 – present
     → 13+ years of weekly issuance, structurally complete.

  B. GMBF + Bonds (opening legs): Jan 2012 – present
     (bonds missing some pre-2011 tap opening legs, but all new-series
     openings from Jan 2011 onward are present)

  C. GMBF + Bonds (both legs, structurally clean): Jan 2015 – present
     → By 2015 the earliest bonds in our dataset (settled Jan 2011,
     maturing Jun 2015) begin paying closing legs, completing the first
     full cycle. New series started from 2011 have both legs in data.

  D. All 3 instruments: Aug 2022 – present

  The most defensible starting point for a combined (GMBF + Bonds)
  settlement flow series is January 2012, with the caveat that bond
  closing legs before June 2015 are absent and pre-2011 bond opening
  legs are missing throughout.
""")


# ─────────────────────────────────────────────────────────────────────────────
# 5.  MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("Settlement Flow Analysis")
    print("=" * 60)

    # Load
    print("\n[1] Loading datasets …")
    df_bills = load_snb_bills()
    df_gmbf  = load_gmbf()
    df_bonds = load_bonds()
    print(f"  SNB Bills : {len(df_bills)} auctions")
    print(f"  GMBF      : {len(df_gmbf)} auctions")
    print(f"  Bonds     : {len(df_bonds)} tranches")

    # Build legs
    print("\n[2] Building settlement legs …")
    df_flows = pd.concat(
        [make_legs(df_bills), make_legs(df_gmbf), make_legs(df_bonds)],
        ignore_index=True,
    ).sort_values(["settlement_date", "instrument", "leg"]).reset_index(drop=True)
    print(f"  Total legs : {len(df_flows)} rows")
    print(f"  Date span  : {df_flows['settlement_date'].min().date()} → "
          f"{df_flows['settlement_date'].max().date()}")

    df_flows.to_csv(FLOWS_OUT, index=False)
    print(f"  Saved → {FLOWS_OUT}")

    # Daily aggregation
    print("\n[3] Daily aggregation …")
    df_daily = daily_aggregate(df_flows)
    df_daily.to_csv(DAILY_OUT, index=False)
    print(f"  {len(df_daily)} daily observations")
    print(f"  Saved → {DAILY_OUT}")

    # Preview
    print("\n  Sample (recent 10 days with activity):")
    recent = df_daily[df_daily["net_chf_mn"] != 0].tail(10)
    print(recent[["date", "opening_chf_mn", "closing_chf_mn", "net_chf_mn",
                   "n_opening", "n_closing"]].to_string(index=False))

    # Coverage report
    coverage_report(df_flows, df_daily)

    return df_flows, df_daily


if __name__ == "__main__":
    df_flows, df_daily = main()
