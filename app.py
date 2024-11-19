import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import sklearn.preprocessing
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
import matplotlib.pyplot as plt

# Page setup
st.title("Stock Trend Prediction App")
st.sidebar.header("User Input")

# Sidebar for stock selection
stock_symbol = st.sidebar.text_input("Enter Stock Symbol (e.g., AAPL, MSFT):", "AAPL")
start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2015-01-01"))
end_date = st.sidebar.date_input("End Date", pd.to_datetime("today"))

# Fetch stock data
@st.cache
def load_data(symbol, start, end):
    data = yf.download(symbol, start=start, end=end)
    data.reset_index(inplace=True)
    return data

data = load_data(stock_symbol, start_date, end_date)

# Display stock data
st.subheader(f"Stock Data for {stock_symbol}")
st.write(data.tail())

# Plot closing price
st.subheader("Closing Price Over Time")
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(data['Date'], data['Close'], label="Closing Price")
ax.set_xlabel("Date")
ax.set_ylabel("Price")
plt.xticks(rotation=45)
st.pyplot(fig)

# Prepare data for LSTM
def preprocess_data(data):
    scaler = sklearn.preprocessing.MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data[['Close']].values)
    
    x_train, y_train = [], []
    for i in range(60, len(scaled_data)):
        x_train.append(scaled_data[i-60:i, 0])
        y_train.append(scaled_data[i, 0])
    
    return np.array(x_train), np.array(y_train), scaler

x_train, y_train, scaler = preprocess_data(data)

# Reshape data
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

# LSTM model
def create_model():
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dense(units=25))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

model = create_model()

# Train the model
if st.sidebar.button("Train Model"):
    with st.spinner("Training model..."):
        model.fit(x_train, y_train, batch_size=32, epochs=5)
    st.success("Model trained successfully!")

# Predict the trend
if st.sidebar.button("Predict"):
    scaled_close = scaler.transform(data[['Close']].values)
    inputs = scaled_close[len(scaled_close) - len(x_train) - 60:]
    
    x_test = []
    for i in range(60, len(inputs)):
        x_test.append(inputs[i-60:i, 0])
    
    x_test = np.array(x_test)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
    
    predictions = model.predict(x_test)
    predictions = scaler.inverse_transform(predictions)
    
    # Visualize predictions
    st.subheader("Predicted vs Actual Closing Price")
    fig2, ax2 = plt.subplots(figsize=(10, 4))
    ax2.plot(data['Date'][len(data)-len(predictions):], data['Close'][len(data)-len(predictions):], label="Actual Price")
    ax2.plot(data['Date'][len(data)-len(predictions):], predictions, label="Predicted Price")
    ax2.legend()
    st.pyplot(fig2)
