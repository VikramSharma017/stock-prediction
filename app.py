import yfinance as yf
import numpy as np
import pandas as pd
import streamlit as st
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

st.title("📈 Stock Price Prediction with LSTM")

# User inputs
ticker = st.text_input("Enter Stock Ticker", value="AAPL").upper()
start_date = st.date_input("Start Date", value=pd.to_datetime("2019-01-01"))
end_date = st.date_input("End Date", value=pd.to_datetime("2024-01-01"))

if st.button("Get Data & Predict"):

    # Load data
    data = yf.download(ticker, start=start_date, end=end_date)
    if data.empty:
        st.error("No data found for this ticker!")
        st.stop()

    st.subheader(f"Raw data for {ticker}")
    st.write(data.tail())

    # Use closing price only
    close_prices = data["Close"].values.reshape(-1, 1)

    # Scale data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(close_prices)

    # Prepare dataset for LSTM (look back 60 days)
    def create_dataset(dataset, look_back=60):
        X, y = [], []
        for i in range(len(dataset) - look_back - 1):
            X.append(dataset[i:(i + look_back), 0])
            y.append(dataset[i + look_back, 0])
        return np.array(X), np.array(y)

    look_back = 60
    X, y = create_dataset(scaled_data, look_back)
    X = X.reshape(X.shape[0], X.shape[1], 1)

    # Split train/test 80/20
    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # Build LSTM model
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(look_back, 1)))
    model.add(LSTM(50))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Train model
    with st.spinner("Training model..."):
        model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=0)

    # Predict
    predictions = model.predict(X_test)
    predictions = scaler.inverse_transform(predictions)
    y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))

    # Plot results
    st.subheader("Prediction vs Actual Closing Prices")
    plt.figure(figsize=(10, 5))
    plt.plot(y_test_actual, label='Actual Price')
    plt.plot(predictions, label='Predicted Price')
    plt.xlabel('Days')
    plt.ylabel('Price')
    plt.legend()
    st.pyplot(plt)

    st.success("Prediction complete!")
