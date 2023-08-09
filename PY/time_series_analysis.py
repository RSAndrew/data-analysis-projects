import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.arima.model import ARIMA
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.arima.model import ARIMA
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import yfinance as yf  # A library to fetch stock price data from Yahoo Finance API

# Function to generate dummy stock price data
def generate_dummy_stock_data(start_date, end_date):
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    num_days = len(date_range)
    prices = np.random.uniform(100, 200, size=num_days)
    df = pd.DataFrame({'Date': date_range, 'Close': prices})
    return df

# Define date range for dummy data
start_date = '2022-01-01'
end_date = '2022-12-31'

# Generate dummy stock price data
dummy_stock_data = generate_dummy_stock_data(start_date, end_date)

# Save dummy stock data to Excel file
excel_file = 'dummy_stock_data.xlsx'
dummy_stock_data.to_excel(excel_file, index=False)

# Load the stock price data from an Excel file
# Replace this with your actual API call to fetch real stock data
ticker = 'AAPL'  # Ticker symbol for Apple Inc.
stock_data = yf.download(ticker, start=start_date, end=end_date)


# Load the stock price data from an Excel file
excel_file = 'stock_price_data.xlsx'
df = pd.read_excel(excel_file, index_col='Date', parse_dates=True)

# Preprocess data
df.sort_index(inplace=True)
train_size = int(0.8 * len(df))
train_data, test_data = df[:train_size], df[train_size:]

# Feature scaling
scaler = MinMaxScaler()
train_scaled = scaler.fit_transform(train_data)
test_scaled = scaler.transform(test_data)

# ARIMA Model
model_arima = ARIMA(train_data, order=(5, 1, 0))
model_arima_fit = model_arima.fit()
forecast_arima = model_arima_fit.forecast(steps=len(test_data))

# LSTM Model
X_train, y_train = [], []
X_test, y_test = [], []
for i in range(60, len(train_scaled)):
    X_train.append(train_scaled[i-60:i])
    y_train.append(train_scaled[i])
for i in range(60, len(test_scaled)):
    X_test.append(test_scaled[i-60:i])
    y_test.append(test_scaled[i])
X_train, y_train = np.array(X_train), np.array(y_train)
X_test, y_test = np.array(X_test), np.array(y_test)
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

model_lstm = Sequential()
model_lstm.add(LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
model_lstm.add(LSTM(50, return_sequences=False))
model_lstm.add(Dense(25))
model_lstm.add(Dense(1))
model_lstm.compile(optimizer='adam', loss='mean_squared_error')
model_lstm.fit(X_train, y_train, batch_size=1, epochs=1)

forecast_lstm = model_lstm.predict(X_test)
forecast_lstm = scaler.inverse_transform(forecast_lstm)

# Evaluate ARIMA model
mse_arima = mean_squared_error(test_data, forecast_arima)
print(f"ARIMA Mean Squared Error: {mse_arima:.2f}")

# Evaluate LSTM model
mse_lstm = mean_squared_error(test_data, forecast_lstm)
print(f"LSTM Mean Squared Error: {mse_lstm:.2f}")

# Visualize predictions
plt.figure(figsize=(10, 6))
plt.plot(test_data.index, test_data['Close'], label='Actual')
plt.plot(test_data.index, forecast_arima, label='ARIMA Forecast', color='orange')
plt.plot(test_data.index, forecast_lstm, label='LSTM Forecast', color='green')
plt.title('Stock Price Forecasting')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.show()
