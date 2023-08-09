import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_squared_error

# Generate dummy time series data
np.random.seed(0)
num_periods = 36
t = np.arange(1, num_periods + 1)
sales = np.random.randint(100, 1000, num_periods)

# Create a DataFrame
data = {
    'Period': t,
    'Sales': sales
}
df = pd.DataFrame(data)

# Save the DataFrame to Excel
excel_file = 'sales_data.xlsx'
df.to_excel(excel_file, index=False)

# Perform time series analysis and forecasting
df['Period'] = pd.to_datetime(df['Period'], format='%Y-%m')
df.set_index('Period', inplace=True)
decomposition = seasonal_decompose(df['Sales'], model='additive')
trend = decomposition.trend
seasonal = decomposition.seasonal
residual = decomposition.resid

# Perform Holt-Winters Exponential Smoothing
model = ExponentialSmoothing(df['Sales'], seasonal='add', seasonal_periods=12)
model_fit = model.fit()
forecast = model_fit.forecast(steps=12)

# Plot the original data, trend, seasonal, and forecast
plt.figure(figsize=(10, 6))
plt.plot(df['Sales'], label='Original Data')
plt.plot(trend, label='Trend')
plt.plot(seasonal, label='Seasonal')
plt.plot(forecast, label='Forecast')
plt.legend()
plt.title('Time Series Analysis and Forecasting')
plt.xlabel('Period')
plt.ylabel('Sales')
plt.grid(True)
plt.show()

print('Excel file "sales_data.xlsx" with dummy data generated.')
