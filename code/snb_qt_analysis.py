"""
SNB Quantitative Tightening (QT) Analysis
==========================================
Downloads, processes, and saves data on two SNB liquidity-absorption instruments:

  1. SNB Bills (Schatzwechsel)
     - Auctioned weekly (Thursday) at 11:00–11:30 CET
     - Terms: ~28d, ~84d, ~168d, ~336d
     - Auction type: Variable Rate, American (VR, A)
     - Auction-level data: Aug 2022 – present (from SNB PDF)
     - Monthly outstanding (ES) from SNB balance sheet: Oct 2008 – present

  2. Reverse Repos (liquidity-absorbing repos)
     - Auctioned daily at 09:00–09:10 CET (except policy decision days: 10:30–10:40)
     - Term: primarily 7 days; occasionally 14 days
     - Payment T+2; redemption after term
     - Monthly outstanding (VRGSF) from SNB balance sheet: Apr 1998 – present
     - No public transaction-level database exists

Data sources:
  - SNB Bills PDF:  https://www.snb.ch/public/bills/snbBillsResults.pdf
  - SNB Balance Sheet (monthly cube snbbipo):
      https://data.snb.ch/api/cube/snbbipo/data/csv/en

Author: SNB_MoPo project
"""

import os
import io
import requests
import warnings
import pandas as pd
import numpy as np
import pdfplumber
import re
from datetime import datetime

warnings.filterwarnings("ignore")

# ── paths ────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR  = os.path.join(BASE_DIR, "input", "raw")
PROC_DIR = os.path.join(BASE_DIR, "input", "processed")
os.makedirs(RAW_DIR,  exist_ok=True)
os.makedirs(PROC_DIR, exist_ok=True)

# ── constants ─────────────────────────────────────────────────────────────────
# The SNB periodically restructures its website; try multiple known URL patterns.
# The file is also cached in input/raw/ so download only if missing.
BILLS_PDF_URLS = [
    "https://www.snb.ch/public/bills/snbBillsResults.pdf",
    "https://www.snb.ch/public/bills/snb_bills_results.pdf",
    "https://www.snb.ch/public/bills/SNBBillsResults.pdf",
]
BIPO_CUBE_URL  = "https://data.snb.ch/api/cube/snbbipo/data/csv/en"

BILLS_PDF_RAW  = os.path.join(RAW_DIR,  "snbBillsResults.pdf")
BIPO_CSV_RAW   = os.path.join(RAW_DIR,  "snb_bipo.csv")

BILLS_PROC     = os.path.join(PROC_DIR, "snb_bills_auctions.csv")
BIPO_PROC      = os.path.join(PROC_DIR, "snb_balance_sheet_monthly.csv")


# ─────────────────────────────────────────────────────────────────────────────
# 1.  DOWNLOAD HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def download_file(urls: str | list[str], dest: str, force: bool = False) -> bytes:
    """
    Download from *urls* (first successful) to *dest*.
    Skip download if already cached unless force=True.
    *urls* may be a single URL string or a list of fallback URLs.
    """
    if not force and os.path.exists(dest):
        print(f"  [cache] {os.path.basename(dest)}")
        with open(dest, "rb") as f:
            return f.read()
    if isinstance(urls, str):
        urls = [urls]
    for url in urls:
        try:
            print(f"  [download] {url}")
            resp = requests.get(url, verify=False, timeout=120)
            resp.raise_for_status()
            with open(dest, "wb") as f:
                f.write(resp.content)
            print(f"  -> {os.path.basename(dest)}  ({len(resp.content):,} bytes)")
            return resp.content
        except Exception as e:
            print(f"  [warn] {e}")
    raise RuntimeError(f"Could not download to {dest} from any of: {urls}")


# ─────────────────────────────────────────────────────────────────────────────
# 2.  SNB BILLS AUCTION DATA (PDF)
# ─────────────────────────────────────────────────────────────────────────────

# Column layout of the SNB Bills Results PDF (one row per ISIN per auction):
#   Date | ISIN | Denomination (CHF) | N/A | Term (days) | Payment date
#   | Redemption date | Auction type | Marginal price (%) | Yield (%)
#   | Bids (CHF mn) | Allotment (CHF mn) | Allot. issuer (CHF mn)
#   | Outstanding (CHF mn)
#
# "N" = new issue;  "A" = tap (additional tranche of existing ISIN)
# Auction type "VR, A" = Variable Rate, American allotment

_DATE_PAT = re.compile(r"^\d{2}\.\d{2}\.\d{4}")

def _parse_bills_line(parts: list[str]) -> dict | None:
    """
    Parse a single tokenised line from the bills PDF.

    Column order (after splitting on whitespace):
      [0]  auction_date   DD.MM.YYYY
      [1]  ISIN           CH…
      [2]  denomination   1'000'000
      [3]  issuance_type  N or A
      [4]  term_days      integer
      [5]  payment_date   DD.MM.YYYY
      [6]  redemption_date DD.MM.YYYY
      [7]  auction_type part 1: "VR,"
      [8]  auction_type part 2: "A"   ← "VR, A" splits into two tokens
      [9]  marginal_price_pct
      [10] yield_pct
      [11] bids_chf_mn      (absent for most-recent rows not yet published)
      [12] allotment_chf_mn
      [13] allot_issuer_chf_mn
      [14] outstanding_chf_mn
    """
    if not parts or not _DATE_PAT.match(parts[0]):
        return None
    try:
        def _d(s: str) -> str:
            return datetime.strptime(s.strip(), "%d.%m.%Y").strftime("%Y-%m-%d")
        def _f(s: str) -> float | None:
            s = s.strip().replace("'", "").replace(",", ".")
            try:
                return float(s)
            except ValueError:
                return None

        # Detect and merge the "VR," + "A" split
        # parts[7] is always "VR," (ends with comma) → merge with parts[8]
        if len(parts) >= 9 and parts[7].endswith(","):
            auction_type = parts[7] + " " + parts[8]
            offset = 1   # subsequent indices shift by +1
        else:
            auction_type = parts[7] if len(parts) > 7 else ""
            offset = 0

        row: dict = {
            "auction_date":          _d(parts[0]),
            "isin":                  parts[1].strip(),
            "denomination_chf":      int(parts[2].replace("'", "").replace(",", "")),
            "issuance_type":         parts[3].strip(),   # N or A
            "term_days":             int(parts[4]),
            "payment_date":          _d(parts[5]),
            "redemption_date":       _d(parts[6]),
            "auction_type":          auction_type,
            "marginal_price_pct":    _f(parts[8  + offset]) if len(parts) > 8  + offset else None,
            "yield_pct":             _f(parts[9  + offset]) if len(parts) > 9  + offset else None,
            "bids_chf_mn":           _f(parts[10 + offset]) if len(parts) > 10 + offset else None,
            "allotment_chf_mn":      _f(parts[11 + offset]) if len(parts) > 11 + offset else None,
            "allot_issuer_chf_mn":   _f(parts[12 + offset]) if len(parts) > 12 + offset else None,
            "outstanding_chf_mn":    _f(parts[13 + offset]) if len(parts) > 13 + offset else None,
        }
        return row
    except Exception:
        return None


def parse_bills_pdf(pdf_bytes: bytes) -> pd.DataFrame:
    """
    Extract all auction rows from the SNB Bills Results PDF.

    Returns a DataFrame with one row per (auction_date × ISIN) pair.
    Term buckets (term_days):
        ~7d  (6–8)    : 1-week bills – used only in the early 2022 restart period
        ~28d (26–30)  : 1-month bills
        ~84d (82–86)  : 3-month bills
        ~168d (166–170): 6-month bills
        ~336d (335–337): 12-month bills
    """
    rows = []
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        for page_num, page in enumerate(pdf.pages):
            text = page.extract_text()
            if not text:
                continue
            for line in text.splitlines():
                parts = line.split()
                row = _parse_bills_line(parts)
                if row:
                    rows.append(row)

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    # Coerce dates
    for col in ("auction_date", "payment_date", "redemption_date"):
        df[col] = pd.to_datetime(df[col])

    # Sort descending (most recent first, like the PDF)
    df = df.sort_values("auction_date", ascending=False).reset_index(drop=True)

    # Bucket the term
    def _bucket(td: int) -> str:
        if td <= 10:   return "7d"
        if td <= 35:   return "28d"
        if td <= 100:  return "84d"
        if td <= 200:  return "168d"
        return "336d"
    df["term_bucket"] = df["term_days"].apply(_bucket)

    return df


# ─────────────────────────────────────────────────────────────────────────────
# 3.  SNB BALANCE SHEET (monthly, cube snbbipo)
# ─────────────────────────────────────────────────────────────────────────────
#
# Key series extracted (all CHF mn):
#   ES    – SNB bills outstanding           (first data: Oct 2008)
#   VRGSF – Reverse-repo liabilities (CHF)  (first data: Apr 1998)
#   FRGSF – Repo claims (CHF) [liquidity-providing repos, SNB asset]
#   N     – Sight deposits of domestic banks
#   T0    – Total assets
#   UA    – Banknotes in circulation
#
# Note: VRGSF on the liabilities side = SNB's obligation to return securities
# it sold under repo (i.e. the SNB absorbed CHF liquidity).

_BIPO_SERIES = {
    "ES":    "snb_bills_outstanding_chf_mn",
    "VRGSF": "reverse_repo_liabilities_chf_mn",
    "FRGSF": "repo_claims_chf_mn",
    "N":     "sight_deposits_chf_mn",
    "T0":    "total_assets_chf_mn",
    "UA":    "banknotes_chf_mn",
}


def parse_bipo_cube(csv_bytes: bytes) -> pd.DataFrame:
    """
    Parse the raw snbbipo cube CSV (semicolon-delimited, 3 header rows).

    Returns a wide-format monthly DataFrame indexed by date (YYYY-MM).
    """
    text = csv_bytes.decode("utf-8", errors="replace")
    df_raw = pd.read_csv(io.StringIO(text), sep=";", skiprows=3, header=0)
    df_raw.columns = ["date", "series", "value"]
    df_raw = df_raw.dropna(subset=["series"])

    # Keep only the series we care about
    df_filt = df_raw[df_raw["series"].isin(_BIPO_SERIES)].copy()
    df_wide = df_filt.pivot(index="date", columns="series", values="value").reset_index()

    # Rename columns to descriptive names
    rename = {"date": "date"}
    rename.update(_BIPO_SERIES)
    df_wide = df_wide.rename(columns=rename)

    df_wide = df_wide.sort_values("date").reset_index(drop=True)

    # Convert date to period-end (last day of month) for merging
    df_wide["date_str"] = df_wide["date"]          # keep original YYYY-MM
    df_wide["date"] = pd.to_datetime(
        df_wide["date"].str[:7] + "-01"
    ) + pd.offsets.MonthEnd(0)

    # Derive: total liquidity-absorbing instruments = bills + reverse repos
    if "snb_bills_outstanding_chf_mn" in df_wide.columns and \
       "reverse_repo_liabilities_chf_mn" in df_wide.columns:
        df_wide["total_liq_absorption_chf_mn"] = (
            df_wide["snb_bills_outstanding_chf_mn"].fillna(0)
            + df_wide["reverse_repo_liabilities_chf_mn"].fillna(0)
        )

    return df_wide


# ─────────────────────────────────────────────────────────────────────────────
# 4.  TERM BUCKET AGGREGATION OF BILLS AUCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def aggregate_bills_weekly(df_bills: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate SNB bills auction data to weekly level by term bucket.

    Returns a DataFrame with one row per auction_date × term_bucket,
    with summed allotment_chf_mn and bids_chf_mn.
    """
    if df_bills.empty:
        return df_bills

    agg = (
        df_bills.groupby(["auction_date", "term_bucket"], as_index=False)
        .agg(
            n_tranches=("isin", "count"),
            allotment_chf_mn=("allotment_chf_mn", "sum"),
            bids_chf_mn=("bids_chf_mn", "sum"),
            outstanding_chf_mn=("outstanding_chf_mn", "sum"),
            avg_yield_pct=("yield_pct", "mean"),
            avg_marginal_price_pct=("marginal_price_pct", "mean"),
        )
    )
    agg = agg.sort_values("auction_date", ascending=False).reset_index(drop=True)
    return agg


def weekly_bills_wide(df_bills: pd.DataFrame) -> pd.DataFrame:
    """
    Pivot weekly bills to wide format: one row per auction date,
    columns = allotment per term bucket (CHF mn).
    """
    if df_bills.empty:
        return df_bills

    agg = aggregate_bills_weekly(df_bills)
    pivot = agg.pivot(
        index="auction_date",
        columns="term_bucket",
        values="allotment_chf_mn"
    ).fillna(0).reset_index()
    pivot.columns.name = None

    # Ensure all standard term columns exist
    for col in ["7d", "28d", "84d", "168d", "336d"]:
        if col not in pivot.columns:
            pivot[col] = 0.0

    # Total allotment
    bucket_cols = [c for c in ["7d", "28d", "84d", "168d", "336d"] if c in pivot.columns]
    pivot["total_allotment_chf_mn"] = pivot[bucket_cols].sum(axis=1)

    return pivot.sort_values("auction_date", ascending=False).reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# 5.  MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 60)
    print("SNB QT Analysis")
    print("=" * 60)

    # ── 5a. SNB Bills ─────────────────────────────────────────────────────────
    print("\n[1] SNB Bills auction data")
    bills_bytes = download_file(BILLS_PDF_URLS, BILLS_PDF_RAW)
    df_bills = parse_bills_pdf(bills_bytes)

    print(f"    Parsed {len(df_bills)} auction rows")
    print(f"    Date range: {df_bills['auction_date'].min().date()} "
          f"to {df_bills['auction_date'].max().date()}")
    print(f"    Term buckets: {df_bills['term_bucket'].value_counts().to_dict()}")

    df_bills.to_csv(BILLS_PROC, index=False)
    print(f"    Saved -> {BILLS_PROC}")

    # Weekly aggregation
    df_bills_weekly = weekly_bills_wide(df_bills)
    bills_weekly_path = os.path.join(PROC_DIR, "snb_bills_weekly.csv")
    df_bills_weekly.to_csv(bills_weekly_path, index=False)
    print(f"    Weekly agg -> {bills_weekly_path}")

    # ── 5b. Balance sheet ─────────────────────────────────────────────────────
    print("\n[2] SNB Balance sheet (cube snbbipo)")
    bipo_bytes = download_file(BIPO_CUBE_URL, BIPO_CSV_RAW)
    df_bipo = parse_bipo_cube(bipo_bytes)

    print(f"    Parsed {len(df_bipo)} monthly observations")
    print(f"    Date range: {df_bipo['date'].min().date()} "
          f"to {df_bipo['date'].max().date()}")

    # Show recent QT period
    print("\n    Recent QT snapshot (last 12 months):")
    cols_show = [
        "date_str",
        "snb_bills_outstanding_chf_mn",
        "reverse_repo_liabilities_chf_mn",
        "total_liq_absorption_chf_mn",
        "sight_deposits_chf_mn",
    ]
    cols_show = [c for c in cols_show if c in df_bipo.columns]
    print(df_bipo[cols_show].tail(12).to_string(index=False))

    df_bipo.to_csv(BIPO_PROC, index=False)
    print(f"\n    Saved -> {BIPO_PROC}")

    # ── 5c. Summary ───────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Done. Output files:")
    for f in [BILLS_PROC, bills_weekly_path, BIPO_PROC]:
        print(f"  {f}")
    print("=" * 60)

    return df_bills, df_bills_weekly, df_bipo


if __name__ == "__main__":
    df_bills, df_bills_weekly, df_bipo = main()
