import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import sklearn.preprocessing
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, LSTM
import matplotlib.pyplot as plt
import seaborn as sns
import requests
from sklearn.metrics import mean_absolute_error
import os
from datetime import datetime

# Page setup
st.title("Stock AI Trading and Analysis App")
st.sidebar.header("User Input")

# Sidebar for stock selection
stock_symbols = st.sidebar.text_input("Enter Stock Symbols (comma-separated, e.g., AAPL, MSFT):", "AAPL, MSFT")
start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2015-01-01"))
end_date = st.sidebar.date_input("End Date", pd.to_datetime("today"))
prediction_horizon = st.sidebar.slider("Prediction Horizon (days):", min_value=1, max_value=30, value=5)

# Fetch stock data
@st.cache_data
def load_data(symbols, start, end):
    data = {}
    for symbol in symbols.split(','):
        symbol = symbol.strip()
        stock_data = yf.download(symbol, start=start, end=end)
        stock_data.reset_index(inplace=True)
        data[symbol] = stock_data
    return data

data = load_data(stock_symbols, start_date, end_date)

# Display stock data
for symbol, stock_data in data.items():
    st.subheader(f"Stock Data for {symbol}")
    st.write(stock_data.tail())

# Add Technical Indicators
def add_technical_indicators(data):
    data['RSI'] = compute_rsi(data['Close'])
    data['MACD'], data['Signal Line'] = compute_macd(data['Close'])
    return data

def compute_rsi(prices, period=14):
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def compute_macd(prices, fast=12, slow=26, signal=9):
    ema_fast = prices.ewm(span=fast, adjust=False).mean()
    ema_slow = prices.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    return macd, signal_line

# Apply indicators to each stock
data = {symbol: add_technical_indicators(stock_data) for symbol, stock_data in data.items()}

# Plot stock prices and indicators
for symbol, stock_data in data.items():
    st.subheader(f"Closing Price and Indicators for {symbol}")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(stock_data['Date'], stock_data['Close'], label="Closing Price")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    plt.xticks(rotation=45)
    plt.legend()
    st.pyplot(fig)

    # Plot RSI
    st.subheader(f"Relative Strength Index (RSI) for {symbol}")
    fig_rsi, ax_rsi = plt.subplots(figsize=(10, 4))
    ax_rsi.plot(stock_data['Date'], stock_data['RSI'], label="RSI", color='orange')
    ax_rsi.axhline(70, linestyle='--', color='red', label="Overbought")
    ax_rsi.axhline(30, linestyle='--', color='green', label="Oversold")
    ax_rsi.set_xlabel("Date")
    ax_rsi.set_ylabel("RSI")
    plt.legend()
    st.pyplot(fig_rsi)

    # Plot MACD
    st.subheader(f"MACD for {symbol}")
    fig_macd, ax_macd = plt.subplots(figsize=(10, 4))
    ax_macd.plot(stock_data['Date'], stock_data['MACD'], label="MACD", color='blue')
    ax_macd.plot(stock_data['Date'], stock_data['Signal Line'], label="Signal Line", color='red')
    ax_macd.axhline(0, linestyle='--', color='black')
    ax_macd.set_xlabel("Date")
    ax_macd.set_ylabel("MACD")
    plt.legend()
    st.pyplot(fig_macd)

# Correlation Matrix
def plot_correlation_matrix(data):
    st.subheader("Correlation Matrix")
    closing_prices = pd.DataFrame({symbol: stock_data['Close'] for symbol, stock_data in data.items()})
    correlation_matrix = closing_prices.corr()
    fig_corr, ax_corr = plt.subplots(figsize=(10, 6))
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", ax=ax_corr)
    st.pyplot(fig_corr)

plot_correlation_matrix(data)

# Prepare data for LSTM
def preprocess_data(data):
    scaler = sklearn.preprocessing.MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data[['Close']].values)

    x_train, y_train = [], []
    for i in range(60, len(scaled_data) - prediction_horizon):
        x_train.append(scaled_data[i - 60:i, 0])
        y_train.append(scaled_data[i:i + prediction_horizon, 0])

    return np.array(x_train), np.array(y_train), scaler

# Train and predict for each stock
for symbol, stock_data in data.items():
    x_train, y_train, scaler = preprocess_data(stock_data)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    # LSTM model
    def create_model():
        model = Sequential()
        model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
        model.add(LSTM(units=50, return_sequences=False))
        model.add(Dense(units=25))
        model.add(Dense(units=prediction_horizon))
        model.compile(optimizer='adam', loss='mean_squared_error')
        return model

    model_file = f"{symbol}_stock_trend_model.h5"
    if os.path.exists(model_file):
        model = load_model(model_file)
    else:
        model = create_model()

    # Train the model
    if st.sidebar.button(f"Train Model for {symbol}"):
        with st.spinner(f"Training model for {symbol}..."):
            model.fit(x_train, y_train, batch_size=32, epochs=5)
            model.save(model_file)
        st.success(f"Model for {symbol} trained successfully!")

    # Predict the trend
    if st.sidebar.button(f"Predict for {symbol}"):
        scaled_close = scaler.transform(stock_data[['Close']].values)
        inputs = scaled_close[-(60 + prediction_horizon):]

        x_test = []
        for i in range(60, len(inputs) - prediction_horizon):
            x_test.append(inputs[i - 60:i, 0])

        x_test = np.array(x_test)
        x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

        predictions = model.predict(x_test)
        predictions = scaler.inverse_transform(predictions)

        # Visualize predictions
        st.subheader(f"Predicted vs Actual Closing Price for {symbol}")
        fig2, ax2 = plt.subplots(figsize=(10, 4))
        actual = stock_data['Close'].values[-len(predictions):]
        ax2.plot(range(len(actual)), actual, label="Actual Price")
        ax2.plot(range(len(actual)), predictions[:, 0], label="Predicted Price")
        ax2.legend()
        st.pyplot(fig2)

# Portfolio Simulation
def simulate_portfolio(data):
    st.sidebar.subheader("Portfolio Simulation")
    investment = st.sidebar.number_input("Initial Investment ($):", value=10000, step=500)
    weights = st.sidebar.text_input("Weights for each stock (comma-separated):", "0.5, 0.5")

    try:
        weights = [float(w) for w in weights.split(',')]
        symbols = list(data.keys())
        if len(weights) != len(symbols):
            st.error("Number of weights must match the number of stocks.")
            return

        normalized_prices = pd.DataFrame({
            symbol: stock_data['Close'] / stock_data['Close'].iloc[0]
            for symbol, stock_data in data.items()
        })

        portfolio_values = (normalized_prices * weights).sum(axis=1) * investment
        st.subheader("Portfolio Value Over Time")
        fig_portfolio, ax_portfolio = plt.subplots(figsize=(10, 6))
        ax_portfolio.plot(normalized_prices.index, portfolio_values, label="Portfolio Value")
        ax_portfolio.set_xlabel("Date")
        ax_portfolio.set_ylabel("Portfolio Value ($)")
        plt.legend()
        st.pyplot(fig_portfolio)

    except ValueError:
        st.error("Invalid weights. Please enter valid numbers.")

simulate_portfolio(data)
