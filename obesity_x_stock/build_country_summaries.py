import os
import pandas as pd

REGION_TO_COUNTRY = {
    "US": "United States of America",
    "Japan": "Japan",
    "Australia": "Australia",
    "India": "India",
    "UK": "United Kingdom",
    "Singapore": "Singapore",
}


def load_ticker_returns(script_dir, tickers):
    """Load per-ticker yearly stock CSVs and compute YoY % return for each ticker/year."""
    stocks_dir = os.path.join(script_dir, "data", "stocks")
    records = []
    for ticker in tickers:
        path = os.path.join(stocks_dir, f"{ticker}.csv")
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path).sort_values("Year")
        df["Return"] = df["End_Price"].pct_change()
        for _, row in df.iterrows():
            if pd.notna(row["Return"]):
                records.append({"Ticker": ticker, "Year": int(row["Year"]), "Return": row["Return"]})
    return pd.DataFrame(records)


def composite_index(returns_df):
    """Equal-weighted average return per year across tickers, chained into an index (base=100)."""
    if returns_df.empty:
        return pd.DataFrame(columns=["Year", "Avg_Return", "Num_Tickers", "Stock_Index"])

    yearly = (
        returns_df.groupby("Year")
        .agg(Avg_Return=("Return", "mean"), Num_Tickers=("Ticker", "nunique"))
        .reset_index()
        .sort_values("Year")
    )

    index_values = []
    level = 100.0
    for _, row in yearly.iterrows():
        level = level * (1 + row["Avg_Return"])
        index_values.append(level)
    yearly["Stock_Index"] = index_values
    return yearly


def load_obesity(script_dir, filename):
    path = os.path.join(script_dir, "data", "bmi_summaries", filename)
    df = pd.read_csv(path)
    obesity_col = "Prevalence of BMI>=30 kg/m² (obesity)"
    combined = (
        df.groupby("Year")[obesity_col]
        .mean()
        .reset_index()
        .rename(columns={obesity_col: "Obesity_Prevalence"})
    )
    return combined


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    stocks_csv = os.path.join(script_dir, "data", "largest_restaurant_and_fastfood_stocks.csv")
    stocks_df = pd.read_csv(stocks_csv)
    relevant = stocks_df[stocks_df["Region"].isin(REGION_TO_COUNTRY.keys())].copy()
    relevant["Country"] = relevant["Region"].map(REGION_TO_COUNTRY)

    output_dir = os.path.join(script_dir, "data", "country_summaries")
    os.makedirs(output_dir, exist_ok=True)

    country_files = {
        "United States of America": "United States of America.csv",
        "Japan": "Japan.csv",
        "Australia": "Australia.csv",
        "India": "India.csv",
        "United Kingdom": "United Kingdom.csv",
        "Singapore": "Singapore.csv",
    }

    all_returns = []

    for country, bmi_file in country_files.items():
        tickers = relevant.loc[relevant["Country"] == country, "Ticker"].tolist()
        returns_df = load_ticker_returns(script_dir, tickers)
        if not returns_df.empty:
            all_returns.append(returns_df)

        stock_index = composite_index(returns_df)
        obesity = load_obesity(script_dir, bmi_file)

        merged = pd.merge(stock_index, obesity, on="Year", how="inner")
        out_path = os.path.join(output_dir, f"{country}.csv")
        merged.to_csv(out_path, index=False)
        print(f"{country}: {len(tickers)} tickers, {merged.shape[0]} years merged -> {out_path}")

    # Global composite: all tickers across all relevant countries, vs world obesity
    global_returns = pd.concat(all_returns, ignore_index=True) if all_returns else pd.DataFrame()
    global_index = composite_index(global_returns)
    world_obesity = load_obesity(script_dir, "world.csv")
    global_merged = pd.merge(global_index, world_obesity, on="Year", how="inner")
    global_path = os.path.join(output_dir, "Global.csv")
    global_merged.to_csv(global_path, index=False)
    print(f"Global: {global_returns['Ticker'].nunique() if not global_returns.empty else 0} tickers, "
          f"{global_merged.shape[0]} years merged -> {global_path}")


if __name__ == "__main__":
    main()
