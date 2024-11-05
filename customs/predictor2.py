from datetime import datetime
import MetaTrader5 as mt5
# display data on the MetaTrader 5 package
print("MetaTrader5 package author: ",mt5.__author__)
print("MetaTrader5 package version: ",mt5.__version__)

# import the 'pandas' module for displaying data obtained in the tabular form
import pandas as pd
pd.set_option('display.max_columns', 500) # number of columns to be displayed
pd.set_option('display.width', 1500)      # max table width to display

# establish connection to MetaTrader 5 terminal
if not mt5.initialize():
    print("initialize() failed, error code =",mt5.last_error())
    quit()

# get 30 GBPUSDm D1 bars from the current day
rates = mt5.copy_rates_from_pos("ETHUSDm", mt5.TIMEFRAME_M15, 0, 30)

# shut down connection to the MetaTrader 5 terminal
mt5.shutdown()
# display each element of obtained data in a new line
# print("Display obtained data 'as is'")
# for rate in rates:
#     print(rate)

# create DataFrame out of the obtained data
rates_frame = pd.DataFrame(rates)
# convert time in seconds into the datetime format
rates_frame['time']=pd.to_datetime(rates_frame['time'], unit='s')

# display data
# print("\nDisplay dataframe with data")
# print(rates)
# print(rates_frame)
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Convert the list data to a NumPy array for easier manipulation
data = rates_frame

data_array = np.array(data)

# Extract features and target labels
price_changes = np.diff(data_array[:, 4])  # Calculate price change from Close prices
volumes = data_array[1:, 5]  # Use tick volume

# Create a simple feature matrix
window_size = 5  # Use the last 5 data points for features
X = []
y = []

for i in range(len(price_changes) - window_size):
    X.append(np.concatenate((price_changes[i:i+window_size], volumes[i:i+window_size])))
    y.append(1 if price_changes[i+window_size] > 0 else 0)

X = np.array(X)
y = np.array(y)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the Random Forest classifier
rf_classifier = RandomForestClassifier(random_state=42)
rf_classifier.fit(X_train, y_train)

# Make predictions
predictions = rf_classifier.predict(X_test)

# Convert predictions to buy (1) or sell (0) signals
buy_signals = np.where(predictions == 1)
sell_signals = np.where(predictions == 0)

print(f"Buy signals: {buy_signals[0]}")
print(f"Sell signals: {sell_signals[0]}")

# Calculate accuracy
accuracy = accuracy_score(y_test, predictions)
print(predictions)
print(f"Accuracy: {accuracy:.2f}")






