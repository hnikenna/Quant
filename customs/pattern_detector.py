import MetaTrader5 as mt5
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense
import matplotlib.pyplot as plt

# Initialize MetaTrader 5
mt5.initialize()

# Define the symbol, timeframe, and the number of bars to fetch
symbol = "BTCUSDm"
timeframe = mt5.TIMEFRAME_M15
bars = 50000  # Adjust the number of bars as needed

# Fetch historical OHLC data
rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, bars)

# Shutdown MetaTrader 5
mt5.shutdown()

# Convert the data to a DataFrame
df = pd.DataFrame(rates)
df['time'] = pd.to_datetime(df['time'], unit='s')  # Convert timestamp to datetime
df.set_index('time', inplace=True)

# Use the closing prices for prediction
data = df['close'].values.reshape(-1, 1)

# Normalize the data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)

# Prepare the data for training the LSTM model
x_train, y_train = [], []
for i in range(60, len(scaled_data)):
    x_train.append(scaled_data[i-60:i, 0])
    y_train.append(scaled_data[i, 0])
x_train, y_train = np.array(x_train), np.array(y_train)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

# Create and train the LSTM model
model = Sequential()
model.add(LSTM(units=100, return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(LSTM(units=100, return_sequences=False))
model.add(Dense(units=25))
model.add(Dense(units=1))
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(x_train, y_train, epochs=20, batch_size=64)

# Prepare the test data
test_data = scaled_data[len(scaled_data) - len(df['close']) + 60:]
x_test = []
for i in range(60, len(test_data)):
    x_test.append(test_data[i-60:i, 0])
x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

# Make predictions using the LSTM model
predictions = model.predict(x_test)
# print(predictions.shape == (x_test.shape[0], x_test.shape[-1]))
predictions = scaler.inverse_transform(predictions)
# print(predictions)

# Create a new DataFrame for predictions with correct index
predicted_df = pd.DataFrame(predictions, columns=['Predictions'], index=df.index[-len(predictions):])

# Make predictions using the LSTM model, extending slightly into the future
x_future = x_test.copy()
while True:
    # for i in range(future_steps):
    # next_step = model.predict(x_future[-1])
    # print(x_future[-1])
    # print(f'X-Future: {len(x_future[-1])}, {x_future[-1].shape}, {x_future.shape}, {len(x_future[-1].reshape(1, x_test.shape[1], 1))}, {str(x_future[-1].reshape(1, x_test.shape[1], 1))}')

    x_test_sub = np.reshape(x_future[-1], (x_future[-1].shape[0], x_future[-1].shape[1], 1))
    next_step = model.predict(x_test_sub)
    # print(next_step.shape == x_future[-1].shape)
    # print(next_step == x_future[-1])
    # print('XFUTURE:', x_future[-1])
    # print('NEXTSTEP', next_step)
    # quit()
    # next_step = np.reshape(next_step, x_future[-1].shape)
    # print(f'Next step: {len(next_step)}, {next_step.shape}, {next_step}')    #, {len(next_step.reshape(1, x_test.shape[1], 1))}, {str(next_step.reshape(1, x_test.shape[1], 1))}, {len(next_step.reshape(1, 1, 1))}, {str(next_step.reshape(1, 1, 1))}')

    if np.array_equal(next_step, x_future[-1]):
        # print(next_step, x_future[-1])
        break
    x_future = np.vstack([x_future, next_step.reshape(1, x_future[-1].shape[0], x_future[-1].shape[1])])

    # x_future = np.append(x_future, next_step.reshape(1, 1, 1), axis=1)

# Inverse transform the predictions
predictions2 = model.predict(x_future)
# print(predictions2.shape == (x_future.shape[0], x_future.shape[-1]))
future_predictions = scaler.inverse_transform(predictions2)
# print(future_predictions)
# future_predictions = scaler.inverse_transform(x_future[:, -future_steps:].reshape(-1, 1))

# Create a new DataFrame for extended predictions with correct index
future_steps = predictions2.shape[0] - predictions.shape[0]
print(future_steps)
future_index = pd.date_range(df.index[-1], periods=future_steps + 1, freq='15T')[1:]
combined_index = df.index[-len(predictions):].union(future_index)
future_predicted_df = pd.DataFrame(future_predictions, columns=['Future Predictions'], index=combined_index)

# Visualize the results including future predictions
plt.figure(figsize=(16, 8))
plt.title('Model Predictions vs Actual Closing Prices')
plt.xlabel('Date')
plt.ylabel('Closing Price')
plt.plot(df['close'])
plt.plot(predicted_df['Predictions'])
plt.plot(future_predicted_df['Future Predictions'], linestyle='dashed', color='green')
plt.legend(['Actual', 'Predictions', 'Future Predictions'])
plt.show()
