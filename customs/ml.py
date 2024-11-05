# Import necessary libraries
import MetaTrader5 as mt5
import talib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from datetime import datetime

# Connect to MetaTrader 5
mt5.initialize()
symbol = "ETHUSDm"
timeframe = mt5.TIMEFRAME_M1

# Retrieve historical data
rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, 1000)
df = pd.DataFrame(rates)
df['time'] = pd.to_datetime(df['time'], unit='s')

# Feature Engineering
df['SMA_50'] = df['close'].rolling(window=50).mean()
df['SMA_200'] = df['close'].rolling(window=200).mean()
df['RSI'] = talib.RSI(df['close'], timeperiod=14)

# Labeling
df['Target'] = df['close'].shift(-1) < df['close']

# Drop NaN values
df = df.dropna()

# Define features and target variable
X = df[['SMA_50', 'SMA_200', 'RSI']]
y = df['Target']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Machine Learning Model - Random Forest
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Model Evaluation
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy}")

# Live Trading
current_data = mt5.copy_rates_from(symbol, timeframe, 0, 1)
current_price = current_data['close'][0]

# Feature extraction for live prediction
live_features = [current_price, df['SMA_50'].iloc[-1], df['SMA_200'].iloc[-1], df['RSI'].iloc[-1]]
live_prediction = model.predict([live_features])

# Trading Decision
if live_prediction:
    print("Predicted: Buy")
    # Implement code to place a buy order in MetaTrader 5
else:
    print("Predicted: Sell")
    # Implement code to place a sell order in MetaTrader 5

mt5.shutdown()
