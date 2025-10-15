import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import yfinance as yf
import tensorflow as tf
import streamlit as st

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)
from tensorflow.keras import backend as K
K.clear_session()

def technical_analysis(data):
    data['MA50'] = data['Close'].rolling(50).mean()
    data['MA200'] = data['Close'].rolling(200).mean()
    delta = data['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss.replace(0, np.finfo(float).eps)  # Avoid division by zero
    data['RSI'] = 100 - (100 / (1 + rs))
    ema12 = data['Close'].ewm(span=12, adjust=False).mean()
    ema26 = data['Close'].ewm(span=26, adjust=False).mean()
    data['MACD'] = ema12 - ema26
    data['MACD_Signal'] = data['MACD'].ewm(span=9, adjust=False).mean()
    return data.bfill().ffill().fillna(0)

@st.cache_resource
def train_lstm_model(_X_train, _y_train, _X_test, _y_test, input_shape):
    model = Sequential()
    model.add(Input(shape=input_shape))
    model.add(LSTM(100, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(100))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='relu'))
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(_X_train, _y_train, epochs=30, batch_size=32, validation_data=(_X_test, _y_test), verbose=0)
    return model

def predict_close(historical, intraday, next_trading_date, symbol):
    try:
        tech_data = technical_analysis(historical)
        data = tech_data[['Open', 'High', 'Low', 'Close', 'Volume', 'RSI', 'MACD']].values
        
        # Drop rows with NaNs
        data = data[~np.isnan(data).any(axis=1)]
        if len(data) < 20:
            return None, MinMaxScaler()
        
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(data)
        
        sequence_length = 5 if symbol.startswith('^') else 15
        X, y = [], []
        for i in range(sequence_length, len(scaled_data)):
            X.append(scaled_data[i-sequence_length:i, :])
            y.append(scaled_data[i, 3])
        X, y = np.array(X), np.array(y)
        X = np.reshape(X, (X.shape[0], X.shape[1], X.shape[2]))
        
        if len(X) == 0:
            print("LSTM sequence insufficient. Using linear regression fallback.")
            X_reg = tech_data[['RSI', 'MACD']].dropna().tail(50).values
            y_reg = tech_data['Close'].dropna().tail(50).values
            if len(X_reg) > 1:
                reg = LinearRegression().fit(X_reg, y_reg)
                predicted_close = reg.predict(X_reg[-1].reshape(1, -1))[0]
                return max(0, predicted_close), scaler
            else:
                return None, scaler
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Use cached model
        model = train_lstm_model(X_train, y_train, X_test, y_test, (X_train.shape[1], X_train.shape[2]))
        
        tech_intraday = technical_analysis(intraday)
        live_data = tech_intraday.tail(1) if not intraday.empty else tech_data.tail(1)
        
        last_sequence_data = tech_data[['Open', 'High', 'Low', 'Close', 'Volume', 'RSI', 'MACD']].values[-sequence_length:]
        if last_sequence_data.shape[0] < sequence_length:
            padding = np.zeros((sequence_length - last_sequence_data.shape[0], 7))
            last_sequence_data = np.vstack((padding, last_sequence_data))
        
        latest_close = float(live_data['Close'].iloc[0]) if not live_data.empty else float(tech_data['Close'].iloc[-1])
        today_row = np.array([
            float(live_data['Open'].iloc[0]) if not live_data.empty else float(tech_data['Open'].iloc[-1]),
            float(live_data['High'].iloc[0]) if not live_data.empty else float(tech_data['High'].iloc[-1]),
            float(live_data['Low'].iloc[0]) if not live_data.empty else float(tech_data['Low'].iloc[-1]),
            float(latest_close),
            float(live_data['Volume'].iloc[0]) if not live_data.empty else float(tech_data['Volume'].iloc[-1]),
            float(live_data['RSI'].iloc[0]) if not live_data.empty else float(tech_data['RSI'].iloc[-1]),
            float(live_data['MACD'].iloc[0]) if not live_data.empty else float(tech_data['MACD'].iloc[-1])
        ])
        
        inputs = np.vstack((last_sequence_data[-(sequence_length-1):], today_row.reshape(1, -1)))
        inputs = scaler.transform(inputs)
        inputs = np.reshape(inputs, (1, inputs.shape[0], inputs.shape[1]))
        predicted = model.predict(inputs, verbose=0)[0][0]
        dummy_row = np.zeros((1, 7))
        dummy_row[0, 3] = predicted
        predicted_close = scaler.inverse_transform(dummy_row)[0, 3]
        
        # Ensure non-negative
        predicted_close = max(0, predicted_close)
        
        ticker = yf.Ticker(symbol)
        fundamentals = {
            'P/E': ticker.info.get('trailingPE', 'N/A'),
            'EPS': ticker.info.get('trailingEps', 'N/A'),
            'Revenue (Cr)': ticker.info.get('totalRevenue', 'N/A') / 10**7,
            'Profit Margin (%)': ticker.info.get('profitMargins', 'N/A') * 100,
            'Market Cap (Cr)': ticker.info.get('marketCap', 'N/A') / 10**7
        }
        analyst_targets = {
            "COALINDIA.NS": 450.00,
            "ADANIPORTS.NS": 1800.00,
            "APOLLOHOSP.NS": 6800.00,
            "ASIANPAINT.NS": 3400.00,
            "AXISBANK.NS": 1300.00,
            "SBIN.NS": 750.00,
        }
        analyst_target = analyst_targets.get(symbol, 450.00)
        
        # Sentiment-based adjustment
        sentiment_boost = 1.05 if "Positive" else 1.0
        if fundamentals['P/E'] != 'N/A' and float(fundamentals['P/E']) < 15:
            predicted_close = (predicted_close * 0.7 * sentiment_boost) + (analyst_target * 0.3)
        predicted_close = max(0, predicted_close)
        
        return predicted_close, scaler
    except Exception as e:
        print(f"Prediction error: {e}")
        return None, None

if __name__ == "__main__":
    historical = yf.download("SBIN.NS", period="1y", interval="1d", auto_adjust=False)
    intraday = yf.download("SBIN.NS", period="1d", interval="5m", prepost=True, auto_adjust=False)
    from datetime import date
    predicted_close, _ = predict_close(historical, intraday if not intraday.empty else historical.tail(1), date(2025, 10, 16), "SBIN.NS")
    print(f"Predicted Close: ₹{predicted_close:.2f} if not None else 'Failed'")
