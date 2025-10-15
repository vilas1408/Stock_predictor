import streamlit as st
from predict_close import predict_close, technical_analysis
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from pytz import timezone

# Fetch data
@st.cache_data(ttl=300)  # Cache for 5 minutes during market hours
def fetch_data(symbol):
    historical = yf.download(symbol, period="1y", interval="1d", auto_adjust=False)
    intraday = yf.download(symbol, period="1d", interval="5m", prepost=True, auto_adjust=False)
    if intraday.empty:
        st.warning("No intraday data available (market closed). Using historical fallback.")
        intraday = historical.tail(1)
    return historical, intraday

# Fundamental Analysis (adjusted for indices)
def fundamental_analysis(ticker, symbol):
    info = ticker.info
    if symbol.startswith('^'):  # Index
        return {
            'P/E': info.get('trailingPE', 'Aggregate P/E ~25 (Nifty 50)'),  # Placeholder for index
            'EPS': 'N/A (Index)',
            'Revenue (Cr)': 'N/A (Index)',
            'Profit Margin (%)': 'N/A (Index)',
            'Market Cap (Cr)': info.get('marketCap', 'N/A (Index Total Cap ~$4.5T)')
        }
    else:
        return {
            'P/E': info.get('trailingPE', 'N/A'),
            'EPS': info.get('trailingEps', 'N/A'),
            'Revenue (Cr)': info.get('totalRevenue', 'N/A') / 10**7,
            'Profit Margin (%)': info.get('profitMargins', 'N/A') * 100,
            'Market Cap (Cr)': info.get('marketCap', 'N/A') / 10**7
        }

# Sentiment Analysis (Placeholder for X API)
def sentiment_analysis(symbol):
    return "Positive"  # Replace with real API

# Main Streamlit App
def main():
    st.title("Stock Price Predictor with Analyst Insights")
    st.write("Predicting closing price using Fundamental, Sentiment, and Technical Analysis.")

    # Sidebar for stock selection
    st.sidebar.header("Settings")
    stock_options = {
        # Nifty 50 Stocks
        "Adani Ports": "ADANIPORTS.NS",
        "Apollo Hospitals": "APOLLOHOSP.NS",
        "Asian Paints": "ASIANPAINT.NS",
        "Axis Bank": "AXISBANK.NS",
        "Bajaj Finserv": "BAJAJFINSV.NS",
        "Bajaj Finance": "BAJFINANCE.NS",
        "Bharti Airtel": "BHARTIARTL.NS",
        "Bharat Petroleum": "BPCL.NS",
        "Britannia Industries": "BRITANNIA.NS",
        "Cipla": "CIPLA.NS",
        "Coal India": "COALINDIA.NS",
        "Divi's Laboratories": "DIVISLAB.NS",
        "Dr. Reddy's Laboratories": "DRREDDY.NS",
        "Eicher Motors": "EICHERMOT.NS",
        "Grasim Industries": "GRASIM.NS",
        "HCL Technologies": "HCLTECH.NS",
        "HDFC Bank": "HDFCBANK.NS",
        "HDFC Life": "HDFCLIFE.NS",
        "Hero MotoCorp": "HEROMOTOCO.NS",
        "Hindalco Industries": "HINDALCO.NS",
        "Hindustan Unilever": "HINDUNILVR.NS",
        "ICICI Bank": "ICICIBANK.NS",
        "ITC": "ITC.NS",
        "JSW Steel": "JSWSTEEL.NS",
        "Kotak Mahindra Bank": "KOTAKBANK.NS",
        "Larsen & Toubro": "LT.NS",
        "LTIMindtree": "LTIM.NS",
        "Mahindra & Mahindra": "M&M.NS",
        "Maruti Suzuki": "MARUTI.NS",
        "Nestle India": "NESTLEIND.NS",
        "NTPC": "NTPC.NS",
        "ONGC": "ONGC.NS",
        "Power Grid": "POWERGRID.NS",
        "Reliance Industries": "RELIANCE.NS",
        "SBI Life": "SBILIFE.NS",
        "State Bank of India": "SBIN.NS",
        "Shriram Finance": "SHRIRAMFIN.NS",
        "Sun Pharma": "SUNPHARMA.NS",
        "Tata Consumer Products": "TATACONSUM.NS",
        "Tata Motors": "TATAMOTORS.NS",
        "Tata Steel": "TATASTEEL.NS",
        "Tech Mahindra": "TECHM.NS",
        "Titan Company": "TITAN.NS",
        "Tata Consultancy Services": "TCS.NS",
        "UltraTech Cement": "ULTRACEMCO.NS",
        "Wipro": "WIPRO.NS",
        # Indices
        "Nifty 50": "^NSEI",
        "Sensex": "^BSESN",
        "Nifty Bank": "^NSEBANK",
        "Nifty IT": "^CNXIT",
        "Nifty Midcap 100": "^CNXMIDCAP",
        # Additional Popular NSE Stocks (Outside Nifty 50)
        "Hindustan Zinc": "HINDZINC.NS",
        "Trent": "TRENT.NS",
        "Samvardhana Motherson": "MOTHERSUMI.NS",
        "Varun Beverages": "VBL.NS",
        "Dixon Technologies": "DIXON.NS",
        "Persistent Systems": "PERSISTENT.NS",
        "KPIT Technologies": "KPITTECH.NS",
        "Affle India": "AFFLE.NS",
        "Sobha Ltd": "SOBHA.NS",
        "Prestige Estates": "PRESTIGE.NS",
        "Godrej Properties": "GODREJPROP.NS",
        "DLF": "DLF.NS",
        "Macrotech Developers": "LODHA.NS",
        "Phoenix Mills": "PHOENIXLTD.NS"
    }
    selected_stock = st.sidebar.selectbox("Select Stock", options=list(stock_options.keys()), index=0)
    symbol = stock_options[selected_stock]

    # Custom search bar for other stocks
    st.sidebar.subheader("Or Search for Other Stocks")
    search_query = st.sidebar.text_input("Enter Stock Name (e.g., 'Infosys')", "")
    if search_query:
        # Simulate search (in real app, use web_search tool here)
        # For demo, map common names to symbols
        search_map = {
            "Infosys": "INFY.NS",
            "Tata Chemicals": "TATACHEM.NS",
            "HDFC AMC": "HDFCAMC.NS",
            "Tata Power": "TATAPOWER.NS",
            "Bharat Electronics": "BEL.NS",
            # Add more mappings as needed
        }
        symbol = search_map.get(search_query, search_query.upper() + ".NS")
        selected_stock = search_query
        st.sidebar.write(f"Found: {symbol}")

    historical, intraday = fetch_data(symbol)
    ticker = yf.Ticker(symbol)

    if historical.empty:
        st.error(f"No historical data available for {symbol}. Check symbol format (e.g., INFY.NS).")
        return

    # Analyses
    fundamentals = fundamental_analysis(ticker, symbol)
    sentiment = sentiment_analysis(symbol)
    tech_data = technical_analysis(historical)

    st.subheader("Fundamental Analysis")
    st.json(fundamentals)

    st.subheader("Sentiment Analysis")
    st.write(f"Market Sentiment: {sentiment}")

    st.subheader("Technical Analysis (Latest)")
    st.write(tech_data.tail(1))

    # Prediction
    ist = timezone('Asia/Kolkata')
    current_date = datetime.now(ist).date()
    predicted_date = current_date
    if current_date.weekday() >= 5 or datetime.now(ist).time() > datetime.strptime("15:30", "%H:%M").time():
        days_to_add = 0
        while (predicted_date.weekday() + days_to_add) % 7 in [5, 6]:
            days_to_add += 1
        predicted_date = current_date + timedelta(days=days_to_add)

    predicted_close, _ = predict_close(historical, intraday, predicted_date, symbol)
    current_price = float(historical['Close'].iloc[-1]) if intraday.empty else float(intraday['Close'].iloc[-1])

    st.subheader(f"Stock: {selected_stock} ({symbol})")
    st.write(f"**Prediction Date:** {predicted_date}")
    st.write(f"**Current Price (as of last update):** ₹{current_price:.2f}")
    if predicted_close is not None:
        st.write(f"**Predicted Closing Price (LSTM + Analyst Adjustment):** ₹{predicted_close:.2f}")
        percentage_change = ((predicted_close - current_price) / current_price) * 100
        st.write(f"**Projected Change:** {percentage_change:.2f}%")
    else:
        st.write(f"**Prediction failed. Using current price as fallback:** ₹{current_price:.2f}")

    st.line_chart(historical['Close'].tail(30))

    st.write("Note: Predictions are for educational purposes. Past performance does not guarantee future results.")

if __name__ == "__main__":
    main()
