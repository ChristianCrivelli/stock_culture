"""
Correlation Analysis: Fast-food/restaurant stock composites vs. obesity prevalence
by country, 1980-2024.

Methodology notes (important for interpretation):
- Both Stock_Index and Obesity_Prevalence are strongly trending series (obesity
  prevalence rises almost monotonically in every country; stock indices compound
  upward over decades). Correlating raw LEVELS of two trending series will produce
  a very high correlation almost by construction ("spurious regression" /
  nonsense correlation from shared trend) - this is NOT evidence of a real
  relationship.
- To get at something meaningful, we look at:
    (1) Level correlation (log stock index vs obesity %) - reported but flagged
        as likely spurious due to shared trend.
    (2) First-differenced / YoY change correlation (Avg_Return vs YoY obesity
        prevalence change) - removes the shared trend, more honest signal.
    (3) Lagged cross-correlation on the differenced series - does stock
        return in year t correlate with obesity change in year t+k for
        k = -3..+3? Tests whether one leads the other.
- Sample sizes are small (13-45 annual observations per country), so p-values
  should be read as indicative, not confirmatory.
"""

import os
import numpy as np
import pandas as pd
from scipy import stats

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "country_summaries")
COUNTRIES = ["Global", "United States of America", "Japan", "Australia",
             "United Kingdom", "Singapore", "India"]


def load(country):
    df = pd.read_csv(os.path.join(DATA_DIR, f"{country}.csv")).sort_values("Year").reset_index(drop=True)
    df["log_stock"] = np.log(df["Stock_Index"])
    df["obesity_pct"] = df["Obesity_Prevalence"] * 100
    # YoY change in obesity prevalence (percentage points)
    df["obesity_yoy_pp"] = df["obesity_pct"].diff()
    # stock return: use Avg_Return if present, else derive from Stock_Index
    if "Avg_Return" in df.columns:
        df["stock_return"] = df["Avg_Return"]
    else:
        df["stock_return"] = df["Stock_Index"].pct_change()
    return df


def level_correlation(df):
    sub = df.dropna(subset=["log_stock", "obesity_pct"])
    r, p = stats.pearsonr(sub["log_stock"], sub["obesity_pct"])
    return r, p, len(sub)


def diff_correlation(df):
    sub = df.dropna(subset=["stock_return", "obesity_yoy_pp"])
    if len(sub) < 3:
        return np.nan, np.nan, len(sub)
    r, p = stats.pearsonr(sub["stock_return"], sub["obesity_yoy_pp"])
    return r, p, len(sub)


def lagged_correlation(df, max_lag=3):
    """
    Positive lag k: stock_return in year t vs obesity_yoy_pp in year t+k
    (stock leads obesity by k years).
    Negative lag k: obesity leads stock by |k| years.
    """
    sub = df.dropna(subset=["stock_return", "obesity_yoy_pp"]).set_index("Year")
    results = []
    for k in range(-max_lag, max_lag + 1):
        s = sub["stock_return"]
        o = sub["obesity_yoy_pp"].shift(-k)  # shift obesity back by k so it aligns with stock at t
        pair = pd.concat([s, o], axis=1, keys=["stock_return", "obesity_future"]).dropna()
        if len(pair) < 4:
            results.append((k, np.nan, np.nan, len(pair)))
            continue
        r, p = stats.pearsonr(pair["stock_return"], pair["obesity_future"])
        results.append((k, r, p, len(pair)))
    return pd.DataFrame(results, columns=["lag_years", "r", "p", "n"])


def main():
    print("=" * 78)
    print("SUMMARY: level correlation (log stock index vs obesity %) — LIKELY SPURIOUS")
    print("=" * 78)
    level_rows = []
    for c in COUNTRIES:
        df = load(c)
        r, p, n = level_correlation(df)
        level_rows.append({"Country": c, "r_level": round(r, 3), "p_level": round(p, 4), "n": n,
                            "years": f"{int(df.Year.min())}-{int(df.Year.max())}"})
    level_df = pd.DataFrame(level_rows)
    print(level_df.to_string(index=False))

    print()
    print("=" * 78)
    print("SUMMARY: YoY differenced correlation (stock return vs obesity pp change)")
    print("=" * 78)
    diff_rows = []
    for c in COUNTRIES:
        df = load(c)
        r, p, n = diff_correlation(df)
        diff_rows.append({"Country": c, "r_diff": round(r, 3) if pd.notna(r) else np.nan,
                           "p_diff": round(p, 4) if pd.notna(p) else np.nan, "n": n})
    diff_df = pd.DataFrame(diff_rows)
    print(diff_df.to_string(index=False))

    print()
    print("=" * 78)
    print("LAGGED CROSS-CORRELATION (differenced series), Global + US")
    print("positive lag = stock return leads obesity change by k years")
    print("negative lag = obesity change leads stock return by k years")
    print("=" * 78)
    for c in ["Global", "United States of America"]:
        df = load(c)
        lag_df = lagged_correlation(df, max_lag=3)
        print(f"\n--- {c} ---")
        print(lag_df.to_string(index=False))

    # Save merged output for later use (e.g. by a viz)
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    level_df.merge(diff_df, on=["Country", "n"], how="outer").to_csv(
        os.path.join(out_dir, "correlation_summary.csv"), index=False)
    print(f"\nSaved summary table to {os.path.join(out_dir, 'correlation_summary.csv')}")


if __name__ == "__main__":
    main()