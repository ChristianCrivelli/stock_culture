"""
Fetches daily OHLC data for each gaming company since 2005 (or IPO, whichever
is later) - annual granularity (used in obesity_x_stock) is too coarse for an
event study, which needs day-level abnormal returns around specific release
dates.

Run locally (not in a sandboxed/offline environment - needs internet access
to Yahoo Finance via yfinance).
"""

import os
import pandas as pd
import yfinance as yf

COMPANIES_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "game_companies.csv")
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "game_stocks_daily")

# Periods to try in order - some non-US indices reject "max" on Yahoo's backend
PERIOD_FALLBACKS = ["max", "10y", "5y", "2y"]


def export_daily(ticker, out_name=None):
    out_name = out_name or ticker
    print(f"Fetching daily data for {ticker}...")
    df = pd.DataFrame()
    for period in PERIOD_FALLBACKS:
        try:
            df = yf.Ticker(ticker).history(period=period, interval="1d")
        except Exception:
            df = pd.DataFrame()
        if not df.empty:
            if period != PERIOD_FALLBACKS[0]:
                print(f"  (used period='{period}' - 'max' was rejected for this symbol)")
            break
    if df.empty:
        print(f"  No data for {ticker}")
        return None
    df.index = pd.to_datetime(df.index).tz_localize(None)
    df = df.sort_index()
    df["Return"] = df["Close"].pct_change()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, f"{out_name}.csv")
    df.to_csv(out_path)
    print(f"  Saved {len(df)} rows -> {out_path}")
    return df


def main():
    companies = pd.read_csv(COMPANIES_CSV)

    # Sanity check: catch a swapped/mislabeled Ticker column before wasting API calls.
    # Real tickers are short and have no spaces; company names do.
    print(f"Loaded {len(companies)} companies from {COMPANIES_CSV}")
    print(f"Columns found: {list(companies.columns)}")
    bad = companies[companies["Ticker"].astype(str).str.contains(" ")]
    if not bad.empty:
        print("\n!! The 'Ticker' column contains values with spaces, which real tickers never have.")
        print("!! This almost always means the Ticker/Company columns got swapped in the CSV.")
        print("!! Offending rows:")
        print(bad[["Ticker", "Company"]].to_string(index=False))
        print("\nFix data/game_companies.csv (header row should be exactly: "
              "Ticker,Company,Country,Primary_Activity,Notes) and re-run. Aborting.")
        return

    for ticker in companies["Ticker"]:
        try:
            export_daily(ticker)
        except Exception as e:
            print(f"  Failed {ticker}: {e}")

    # Also fetch a market benchmark per exchange for abnormal-return calculation.
    # Each entry is a list of candidate symbols tried in order - some regional
    # indices are unreliable on Yahoo and need a fallback (e.g. an ETF tracking
    # the same market instead of the raw index).
    benchmark_candidates = {
        "S&P 500 (US)": ["^GSPC"],
        "CAC 40 (France - Ubisoft)": ["^FCHI"],
        "Nikkei 225 (Japan)": ["^N225"],
        "KOSPI (South Korea)": ["^KS11"],
        "Poland (CD Projekt)": ["^WIG20", "WIG20.WA", "EPOL"],  # EPOL = iShares MSCI Poland ETF, fallback proxy
    }
    for label, candidates in benchmark_candidates.items():
        saved = False
        for symbol in candidates:
            try:
                df = export_daily(symbol, out_name=candidates[0])  # always save under the primary name
            except Exception as e:
                df = None
                print(f"  Failed benchmark {symbol} ({label}): {e}")
            if df is not None:
                saved = True
                break
        if not saved:
            print(f"  Could not fetch any candidate for benchmark: {label} (tried {candidates})")


if __name__ == "__main__":
    main()