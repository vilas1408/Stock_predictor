# model_builder.py
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from sklearn.preprocessing import MinMaxScaler
import yfinance as yf

def build_lstm_model(historical_data):
    data = historical_data[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)
    
    sequence_length = 15
    X, y = [], []
    for i in range(sequence_length, len(scaled_data)):
        X.append(scaled_data[i-sequence_length:i])
        y.append(scaled_data[i, 3])  # Close price
    X, y = np.array(X), np.array(y)
    
    if len(X) == 0:
        return None, None
    
    model = Sequential()
    model.add(Input(shape=(X.shape[1], X.shape[2])))
    model.add(LSTM(100, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(100))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='relu'))
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X, y, epochs=30, batch_size=32, verbose=0)
    return model, scaler

def predict_close(model, scaler, live_data, historical_data):
    if model is None or scaler is None:
        return None
    try:
        last_15_data = historical_data[['Open', 'High', 'Low', 'Close', 'Volume']].values[-15:]
        if last_15_data.shape[0] < 15:
            padding = np.zeros((15 - last_15_data.shape[0], 5))
            last_15_data = np.vstack((padding, last_15_data))
        
        latest_close = float(live_data['Close'].iloc[-1]) if not live_data.empty else float(historical_data['Close'].iloc[-1])
        today_row = np.array([
            float(live_data['Open'].iloc[-1]) if not live_data.empty else float(historical_data['Open'].iloc[-1]),
            float(live_data['High'].iloc[-1]) if not live_data.empty else float(historical_data['High'].iloc[-1]),
            float(live_data['Low'].iloc[-1]) if not live_data.empty else float(historical_data['Low'].iloc[-1]),
            float(latest_close),
            float(live_data['Volume'].iloc[-1]) if not live_data.empty else float(historical_data['Volume'].iloc[-1])
        ])
        
        inputs = np.vstack((last_15_data[-14:], today_row.reshape(1, -1)))
        inputs = scaler.transform(inputs)
        inputs = np.reshape(inputs, (1, inputs.shape[0], inputs.shape[1]))
        predicted = model.predict(inputs, verbose=0)[0][0]
        dummy_row = np.zeros((1, 5))
        dummy_row[0, 3] = predicted
        predicted_close = scaler.inverse_transform(dummy_row)[0, 3]
        return max(0, predicted_close)
    except Exception as e:
        print(f"Prediction error: {e}")
        return None

def fetch_intraday_data(symbol):
    try:
        return yf.download(symbol, period="1d", interval="5m", prepost=True, auto_adjust=False)
    except Exception as e:
        print(f"Intraday data fetch error: {e}")
        return pd.DataFrame()