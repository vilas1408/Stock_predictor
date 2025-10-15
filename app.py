import streamlit as st
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from pytz import timezone
import time
from predict_close import predict_close

st.title("Stock Price Predictor with Analyst Insights")
st.write("Predicting closing price using Fundamental, Sentiment, and Technical Analysis.")

# Fetch data with corrected yfinance call and robust error handling
@st.cache_data(ttl=300)
def fetch_data(symbol, _cache_buster=0):
    def download_with_retry(symbol, period, interval, retries=5, prepost=False):
        for attempt in range(retries):
            try:
                data = yf.download(symbol, period=period, interval=interval, auto_adjust=False, prepost=prepost)
                if not data.empty:
                    return data
                st.warning(f"Empty data for {symbol} on attempt {attempt + 1}, retrying...")
                time.sleep(2 ** attempt + 1)  # Exponential backoff + 1s
            except Exception as e:
                st.warning(f"Error for {symbol} on attempt {attempt + 1}: {e}, retrying...")
                time.sleep(2 ** attempt + 1)
        st.error(f"Failed to fetch data for {symbol} after {retries} attempts. Check symbol (e.g., INFY.NS) or try later.")
        return None

    historical = download_with_retry(symbol, period="1y", interval="1d")
    intraday = download_with_retry(symbol, period="1d", interval="5m", prepost=True)
    
    if historical is None or historical.empty:
        return None, None
    if intraday is None or intraday.empty:
        st.warning(f"No intraday data for {symbol} (market likely closed). Using historical data as fallback.")
        intraday = historical.tail(1)
    
    return historical, intraday

# Generate reasoning for prediction
def generate_reasoning(predicted_close, current_price, fundamentals, sentiment, latest_tech, stock_symbol):
    reasons = []
    projected_change = ((predicted_close - current_price) / current_price) * 100 if predicted_close else 0
    
    pe_ratio = fundamentals["P/E"]
    pe_text = f"P/E {pe_ratio}" if pe_ratio != "N/A" else "P/E unavailable (index)"
    if pe_ratio != "N/A" and float(pe_ratio) < 15:
        reasons.append(f"**Fundamentals**: Low P/E ({pe_ratio}) suggests undervaluation, triggering a conservative adjustment with 30% weight to analyst targets (~₹750 for {stock_symbol}), pulling the prediction lower despite strong earnings.")
    else:
        reasons.append(f"**Fundamentals**: {pe_text}. Fair valuation supports stability, but macro factors (e.g., interest rate hikes) may limit upside.")

    sentiment_boost = 1.05 if sentiment == "Positive" else 1.0
    reasons.append(f"**Sentiment**: {sentiment} market mood adds a {sentiment_boost*100-100:.0f}% boost to the prediction, reflecting optimism from retail and analyst confidence, though tempered by sector risks (e.g., banking NPAs).")

    rsi = latest_tech["RSI"].iloc[-1]
    macd = latest_tech["MACD"].iloc[-1]
    rsi_status = "overbought (potential correction)" if rsi > 60 else "neutral-bullish"
    macd_status = "bullish momentum" if macd > 0 else "bearish momentum"
    reasons.append(f"**Technicals**: RSI ({rsi:.2f}) indicates {rsi_status}, and MACD ({macd:.2f}) shows {macd_status}. The LSTM model detects a {projected_change:.2f}% change, likely due to {'profit-taking near resistance' if projected_change < 0 else 'continued uptrend'}.")

    reasons.append(f"**Overall**: The prediction ({'₹{:.2f}'.format(predicted_close) if predicted_close else 'failed'}) combines LSTM output (70% weight), analyst targets (30%), and a {sentiment_boost*100-100:.0f}% sentiment boost. {'Bearish due to overbought signals and macro risks.' if projected_change < 0 else 'Bullish due to strong momentum and sentiment.'}")

    return "\n\n".join(reasons)

# Main app
if 'cache_buster' not in st.session_state:
    st.session_state.cache_buster = 0

stock_symbol = st.selectbox(
    "Select a stock or index (NSE symbols)",
    ["^NSEI", "SBIN.NS", "COALINDIA.NS", "ADANIPORTS.NS", "APOLLOHOSP.NS", "ASIANPAINT.NS", "AXISBANK.NS", "PICCADIL.NS"],
    index=1  # Default to SBIN.NS to avoid ADANIPORTS.NS issues
)
today = datetime.today()
prediction_date = st.date_input("Select prediction date", min_value=today, max_value=today + timedelta(days=30))

# Fetch data with fallback
historical, intraday = fetch_data(stock_symbol, st.session_state.cache_buster)
if historical is None:
    st.warning(f"Switching to fallback symbol SBIN.NS due to data fetch failure.")
    stock_symbol = "SBIN.NS"
    st.session_state.cache_buster += 1  # Clear cache
    historical, intraday = fetch_data(stock_symbol, st.session_state.cache_buster)
    if historical is None:
        if st.button("Retry Data Fetch"):
            st.session_state.cache_buster += 1
            st.rerun()
        st.error("All data fetch attempts failed. Please try again later or use a different symbol.")
        st.stop()

# Fundamental analysis
ticker = yf.Ticker(stock_symbol)
fundamentals = {
    "P/E": ticker.info.get("trailingPE", "N/A" if not stock_symbol.startswith("^") else "Aggregate P/E ~25 (Nifty 50)"),
    "EPS": ticker.info.get("trailingEps", "N/A"),
    "Revenue (Cr)": ticker.info.get("totalRevenue", "N/A") / 10**7,
    "Profit Margin (%)": ticker.info.get("profitMargins", "N/A") * 100,
    "Market Cap (Cr)": ticker.info.get("marketCap", "N/A" if not stock_symbol.startswith("^") else "N/A (Index Total Cap ~$4.5T)") / 10**7
}

# Sentiment analysis (simulated)
sentiment = "Positive"

# Technical analysis (latest)
latest_tech = historical.tail(1).copy()
latest_tech["RSI"] = 60.08 if stock_symbol == "SBIN.NS" else 63.36  # Example values
latest_tech["MACD"] = 9.26  # Example value
latest_tech = latest_tech[["Open", "High", "Low", "Close", "Volume", "RSI", "MACD"]]

# Predict closing price
predicted_close, scaler = predict_close(historical, intraday, prediction_date, stock_symbol)
current_price = float(historical["Close"].iloc[-1])

# Display results
st.subheader("Fundamental Analysis")
st.json(fundamentals)

st.subheader("Sentiment Analysis")
st.write(f"Market Sentiment: {sentiment}")

st.subheader("Technical Analysis (Latest)")
st.dataframe(latest_tech)

st.subheader(f"Stock: {stock_symbol}")
st.write(f"Prediction Date: {prediction_date}")
st.write(f"Current Price (as of last update): ₹{current_price:.2f}")

if predicted_close is not None:
    projected_change = ((predicted_close - current_price) / current_price) * 100
    st.write(f"Predicted Closing Price (LSTM + Analyst Adjustment): ₹{predicted_close:.2f}")
    st.write(f"Projected Change: {projected_change:.2f}%")
    st.subheader("Reason for Prediction")
    st.markdown(generate_reasoning(predicted_close, current_price, fundamentals, sentiment, latest_tech, stock_symbol))
else:
    st.write(f"Prediction failed. Using current price as fallback: ₹{current_price:.2f}")
