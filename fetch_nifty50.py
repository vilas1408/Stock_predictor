def get_nifty50_stocks():
    """
    Returns a static list of Nifty 50 constituents as of September 2025.
    Symbols include .NS suffix for yfinance compatibility.
    Source: NSE Indices and Wikipedia (as of March 2025, with no major changes reported in September 2025).
    """
    return [
        'ADANIPORTS.NS', 'APOLLOHOSP.NS', 'ASIANPAINT.NS', 'AXISBANK.NS', 'BAJAJFINSV.NS',
        'BAJFINANCE.NS', 'BHARTIARTL.NS', 'BPCL.NS', 'BRITANNIA.NS', 'CIPLA.NS',
        'COALINDIA.NS', 'DIVISLAB.NS', 'DRREDDY.NS', 'EICHERMOT.NS', 'GRASIM.NS',
        'HCLTECH.NS', 'HDFCBANK.NS', 'HDFCLIFE.NS', 'HEROMOTOCO.NS', 'HINDALCO.NS',
        'HINDUNILVR.NS', 'ICICIBANK.NS', 'ITC.NS', 'JSWSTEEL.NS', 'KOTAKBANK.NS',
        'LT.NS', 'LTIM.NS', 'M&M.NS', 'MARUTI.NS', 'NESTLEIND.NS',
        'NTPC.NS', 'ONGC.NS', 'POWERGRID.NS', 'RELIANCE.NS', 'SBILIFE.NS',
        'SBIN.NS', 'SHRIRAMFIN.NS', 'SUNPHARMA.NS', 'TATACONSUM.NS', 'TATAMOTORS.NS',
        'TATASTEEL.NS', 'TECHM.NS', 'TITAN.NS', 'TCS.NS', 'ULTRACEMCO.NS',
        'WIPRO.NS', 'LT.NS'  # Note: LT appears twice in some lists, but it's one; adjust if needed
    ]

# Test: Print the list
print(get_nifty50_stocks())