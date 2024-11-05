import MetaTrader5 as mt5
from datetime import datetime
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Initialize MetaTrader 5
if not mt5.initialize():
    print("initialize() failed, error code =", mt5.last_error())
    quit()

# Fetch tick data
symbol = "XAUUSDm"
ticks = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M5, 0, 30)

# Shutdown MetaTrader 5
mt5.shutdown()

# create DataFrame out of the obtained data
rates_frame = pd.DataFrame(ticks)
# convert time in seconds into the datetime format
rates_frame['time']=pd.to_datetime(rates_frame['time'], unit='s')

# display data
print("\nDisplay dataframe with data")
print(rates_frame)

# Extract bid close prices
close_prices = np.array([tick[1] for tick in ticks])

# Prepare data for modeling
X = close_prices[:-1].reshape(-1, 1)  # Feature: current close price
y = close_prices[1:]  # Target: next close price

# print(y)
# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train a Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict next close price
latest_close_price = X[-1]
# print(np.array([[latest_close_price]]))
next_predicted_close = model.predict(np.array([[latest_close_price]])[0])
print("Latest open price:", latest_close_price)
print("Last open price:", close_prices[-1])
print("Predicted next open price:", next_predicted_close)


if float(next_predicted_close[0]) > close_prices[-1]:
    print(f'We anticipate that the next open price will be substantially UPPER '
          f'by {round(float(next_predicted_close[0]) - close_prices[-1], 8)} points '
          f'than the previous open price.')
else:
    print(f'We anticipate that the next open price will be much less '
          f'by {round(close_prices[-1] - float(next_predicted_close[0]), 8)} points '
          f'than the previous open price.')

