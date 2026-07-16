import os
import sys
import time
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from export_yearly_stock_summary import export_yearly_stock_summary

# Maps the Region values in largest_restaurant_and_fastfood_stocks.csv to the
# country names used in data/bmi_summaries (only countries we have BMI data for)
REGION_TO_COUNTRY = {
    "US": "United States of America",
    "Japan": "Japan",
    "Australia": "Australia",
    "India": "India",
    "UK": "United Kingdom",
    "Singapore": "Singapore",
}


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    stocks_csv = os.path.join(script_dir, "data", "largest_restaurant_and_fastfood_stocks.csv")
    stocks_df = pd.read_csv(stocks_csv)

    relevant = stocks_df[stocks_df["Region"].isin(REGION_TO_COUNTRY.keys())].copy()
    relevant["Country"] = relevant["Region"].map(REGION_TO_COUNTRY)

    print(f"Found {len(relevant)} tickers across {relevant['Country'].nunique()} relevant countries")

    results = []
    for _, row in relevant.iterrows():
        ticker = row["Ticker"]
        country = row["Country"]
        try:
            df = export_yearly_stock_summary(ticker)
            ok = df is not None and not df.empty
        except Exception as e:
            print(f"Failed to fetch {ticker}: {e}")
            ok = False
        results.append({"Ticker": ticker, "Country": country, "Success": ok})
        time.sleep(0.3)  # be polite to the API

    results_df = pd.DataFrame(results)
    summary_path = os.path.join(script_dir, "data", "stock_fetch_results.csv")
    results_df.to_csv(summary_path, index=False)

    print("\n=== Fetch summary ===")
    print(results_df.groupby("Country")["Success"].agg(["sum", "count"]))
    failed = results_df[~results_df["Success"]]
    if not failed.empty:
        print("\nFailed tickers:")
        print(failed[["Ticker", "Country"]].to_string(index=False))


if __name__ == "__main__":
    main()
