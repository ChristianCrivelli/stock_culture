"""
Event study: for each game release in game_release_calendar.csv, compute the
cumulative abnormal return (CAR) of the publisher's stock in a window around
the release date, where "abnormal" means the stock's return minus its local
market benchmark's return on the same day (removes general market movement,
same principle as the excess-return adjustment used in obesity_x_stock).

Window: [-5, +20] trading days around release (release day = day 0). Pre-
release window captures anticipation/hype; post-release window captures the
market's reaction to actual sales/reviews.

Run after export_daily_stock_prices.py has populated data/game_stocks_daily/.
"""

import os
import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
DAILY_DIR = os.path.join(BASE, "data", "game_stocks_daily")
CALENDAR_PATH = os.path.join(BASE, "data", "game_release_calendar.csv")

# Which local benchmark to use per ticker (must match tickers fetched in
# export_daily_stock_prices.py)
BENCHMARK_FOR_TICKER = {
    "TTWO": "^GSPC", "EA": "^GSPC", "ATVI": "^GSPC",
    "UBI.PA": "^FCHI",
    "CDR.WA": "WIG20.WA",
    "7974.T": "^N225", "9697.T": "^N225", "9684.T": "^N225", "7832.T": "^N225",
    "259960.KS": "^KS11",
}

PRE_WINDOW = 5    # trading days before release
POST_WINDOW = 20  # trading days after release


def load_returns(ticker):
    path = os.path.join(DAILY_DIR, f"{ticker}.csv")
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    return df["Return"]


def car_for_event(stock_returns, bench_returns, event_date):
    event_date = pd.Timestamp(event_date)
    # Align on trading days that exist for the stock
    idx = stock_returns.index
    if event_date not in idx:
        # snap to next available trading day
        future = idx[idx >= event_date]
        if len(future) == 0:
            return None
        event_date = future[0]
    pos = idx.get_loc(event_date)
    start = max(pos - PRE_WINDOW, 0)
    end = min(pos + POST_WINDOW, len(idx) - 1)
    window_idx = idx[start:end + 1]

    abnormal = (stock_returns.reindex(window_idx) - bench_returns.reindex(window_idx)).fillna(0)
    car = (1 + abnormal).cumprod() - 1
    return pd.Series(car.values, index=range(start - pos, end - pos + 1))  # relative day index


def main():
    calendar = pd.read_csv(CALENDAR_PATH, parse_dates=["Release_Date"])
    all_results = []

    for _, row in calendar.iterrows():
        ticker = row["Ticker"]
        bench_ticker = BENCHMARK_FOR_TICKER.get(ticker)
        if bench_ticker is None:
            print(f"No benchmark mapped for {ticker}, skipping {row['Title']}")
            continue
        try:
            stock_ret = load_returns(ticker)
            bench_ret = load_returns(bench_ticker)
        except FileNotFoundError:
            print(f"Missing daily data for {ticker} or {bench_ticker} - run export_daily_stock_prices.py first")
            continue

        car = car_for_event(stock_ret, bench_ret, row["Release_Date"])
        if car is None:
            continue
        final_car = car.iloc[-1]
        all_results.append({
            "Ticker": ticker, "Company": row["Company"], "Title": row["Title"],
            "Release_Date": row["Release_Date"].date(), "Tier": row["Tier"],
            f"CAR_day{-PRE_WINDOW}_to_day{POST_WINDOW}": round(final_car, 4),
        })

    results_df = pd.DataFrame(all_results)
    print(results_df.to_string(index=False))

    out_path = os.path.join(BASE, "data", "event_study_results.csv")
    results_df.to_csv(out_path, index=False)
    print(f"\nSaved event study results -> {out_path}")

    # Summary by tier: does "Major" tier releases show bigger abnormal returns than "Minor"?
    car_col = [c for c in results_df.columns if c.startswith("CAR_")][0]
    print("\nMean CAR by release tier:")
    print(results_df.groupby("Tier")[car_col].agg(["mean", "std", "count"]))


if __name__ == "__main__":
    main()
