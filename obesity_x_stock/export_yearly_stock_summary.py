import os
import pandas as pd
import yfinance as yf

def export_yearly_stock_summary(ticker_symbol):
    """
    Fetches full historical data for a given stock ticker, calculates yearly metrics,
    and exports the result to a CSV file at ./data/stocks/{ticker}.csv
    """
    print(f"Fetching data for {ticker_symbol}...")
    
    # Retrieve complete historical dataset since creation
    df = yf.Ticker(ticker_symbol).history(period="max")
    
    if df.empty:
        print(f"No data found for ticker: {ticker_symbol}")
        return None

    # Sort index chronologically and parse to standard Datetime format
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    
    # Grouping by calendar year
    df['Year'] = df.index.year
    
    yearly_records = []
    for year, group in df.groupby('Year'):
        group = group.sort_index()
        
        # Calculate parameters
        avg_price = group['Close'].mean()
        start_price = group['Close'].iloc[0]
        end_price = group['Close'].iloc[-1]
        
        yearly_records.append({
            'Year': year,
            'Average_Price': avg_price,
            'Start_Price': start_price,
            'End_Price': end_price
        })
        
    yearly_df = pd.DataFrame(yearly_records)
    
    # Compute differences from the previous year
    yearly_df['YoY_End_Price_Diff'] = yearly_df['End_Price'].diff()
    yearly_df['YoY_Avg_Price_Diff'] = yearly_df['Average_Price'].diff()
    
    # Define and ensure directory exists
    output_dir = "obesity_x_stock/data/stocks"
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the file to the specific folder structure
    output_path = os.path.join(output_dir, f"{ticker_symbol}.csv")
    yearly_df.to_csv(output_path, index=False)
    
    print(f"Successfully exported data to: {output_path}")
    return yearly_df

if __name__ == "__main__":
    export_yearly_stock_summary("MCD")