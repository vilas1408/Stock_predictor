import yfinance as yf

def fetch_data(stock_symbol, period='1y', interval='1d', live=False):
    """
    Fetch historical or live intraday data for a given stock symbol.
    
    Args:
        stock_symbol (str): Stock symbol (e.g., '^NSEI' for Nifty 50 Index, 'RELIANCE.NS' for Reliance).
        period (str): Time period for historical data (default: '1y' for 1 year).
        interval (str): Data interval (default: '1d' for daily, '1m' for minute if live).
        live (bool): If True, fetch today's intraday data (default: False for historical).
    
    Returns:
        pandas.DataFrame or pandas.Series: Data for historical, or a row for live data.
    
    Raises:
        ValueError: If no data is available (e.g., market closed or API issue).
    """
    # Fetch data with auto_adjust=False to suppress FutureWarning
    data = yf.download(stock_symbol, period='1d' if live else period, 
                      interval='1m' if live else interval, auto_adjust=False)
    
    # Debug: Print downloaded data shape
    print(f"Downloaded data shape: {data.shape if not data.empty else 'Empty'}")
    
    if live:
        if not data.empty:
            return data.iloc[0]  # Return first row (approx 9:15 AM data) with Open, High, Low, Volume
        else:
            raise ValueError("No live data available. Market might be closed or API issue.")
    else:
        if data.empty:
            raise ValueError(f"No historical data available for {stock_symbol}.")
        return data

# Test function (uncomment to test independently)
# if __name__ == "__main__":
#     print(fetch_data('^NSEI', period='5d').tail())
#     try:
#         live_data = fetch_data('^NSEI', live=True)
#         print("Live data:", live_data)
#     except ValueError as e:
#         print(f"Live data error: {e}")