import pandas as pd
import yfinance as yf

def get_calendar_rolling_data(ticker_symbol):
    print(f"Fetching data for {ticker_symbol}...")
    df = yf.Ticker(ticker_symbol).history(period="10y")
    
    if df.empty:
        print("No data found.")
        return None

    # Ensure the index is a DatetimeIndex (required for time-string rolling windows)
    df.index = pd.to_datetime(df.index)
    
    # Sort index just to be absolutely sure data is in chronological order
    df = df.sort_index()
    
    # 5 calendar years = (365 days * 5) + 1 leap day = 1826 days
    calendar_window = '1826D'
    
    print("Calculating 5 calendar-year rolling metrics...")
    df['5Y_Rolling_Avg_Close'] = df['Close'].rolling(window=calendar_window).mean()
    df['5Y_Rolling_Max'] = df['Close'].rolling(window=calendar_window).max()
    df['5Y_Rolling_Min'] = df['Close'].rolling(window=calendar_window).min()
    
    # Reset index so 'Date' becomes a standard column before saving
    df = df.reset_index()
    
    # Save to CSV
    output_filename = f"{ticker_symbol}.csv"
    df.to_csv(output_filename, index=False)
    print(f"Successfully saved calendar-based rolling data to {output_filename}")
    return df

# Example usage:
get_calendar_rolling_data("AAPL")