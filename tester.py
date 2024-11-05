import MetaTrader5 as mt5
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Connect to MetaTrader 5
mt5.initialize()

# Load historical price data (replace 'BTCUSDm' and 'M1' accordingly)
symbol = 'BTCUSDm'
timeframe = mt5.TIMEFRAME_H1
prices = mt5.copy_rates_from_pos(symbol, timeframe, 0, 50)

# Extract 'close' prices from the structured array
close_prices = prices['close']

# Create a DataFrame for easier manipulation
history = mt5.copy_rates_from_pos(symbol, timeframe, 0, 100)

sticks = 26
while True:
    sticks += 1
    macd_ticks = mt5.copy_rates_from_pos(symbol, timeframe, 0, sticks)

    # MACD
    macd_df = pd.DataFrame(macd_ticks)

    # Calculate MACD manually
    macd_df['ema_fast'] = macd_df['close'].ewm(span=12, adjust=False).mean()
    macd_df['ema_slow'] = macd_df['close'].ewm(span=26, adjust=False).mean()
    macd_df['macd'] = macd_df['ema_fast'] - macd_df['ema_slow']
    macd_df['signal'] = macd_df['macd'].ewm(span=9, adjust=False).mean()

    # Print the DataFrame with MACD values
    # print(macd_df[['time', 'macd', 'signal']])
    macd = macd_df['macd'].iloc[-1]
    signal = macd_df['signal'].iloc[-1]

    if 30 < abs(macd - signal) < 51:
        print('Stick:', sticks)
        print(macd, signal)
        quit()
    else:
        print('.', end='')

# macd_values.append(macd)
# macd_signal_values.append(signal)
quit()
# Convert data to DataFrame
df = pd.DataFrame(history)

# Calculate Exponential Moving Averages (EMA)
# Convert data to DataFrame
ema_df = pd.DataFrame(history)

# Calculate Exponential Moving Averages (EMA)
# Determine short and long window scaling based on the timeframe
scaling_factor_short = 9
scaling_factor_long = 12

# Calculate dynamic short and long windows
short_window = min(int(timeframe * scaling_factor_short), 50)
long_window = min(int(timeframe * scaling_factor_long), 200)

ema_df['short_ema'] = ema_df['close'].ewm(span=short_window, adjust=False).mean()
ema_df['long_ema'] = ema_df['close'].ewm(span=long_window, adjust=False).mean()

# Generate buy/sell signals using loc to avoid the SettingWithCopyWarning
ema_df.loc[short_window:, 'signal'] = 0  # 0 represents no action
ema_df.loc[short_window:, 'signal'] = np.where(ema_df['short_ema'][short_window:] > ema_df['long_ema'][short_window:], 1, 0)  # Buy signal
ema_df.loc[short_window:, 'signal'] = np.where(ema_df['short_ema'][short_window:] < ema_df['long_ema'][short_window:], -1, ema_df['signal'][short_window:])  # Sell signal

# Print the DataFrame with signals
print(ema_df[['time', 'close', 'short_ema', 'long_ema', 'signal']])

quit()
df = pd.DataFrame({'close': close_prices})
# Define moving averages
short_window = 20
long_window = 50

# Calculate moving averages using pandas
df['short_ma'] = df['close'].rolling(window=short_window).mean()
df['long_ma'] = df['close'].rolling(window=long_window).mean()

# Generate signals
df['signals'] = np.where(df['short_ma'] > df['long_ma'], 'Buy', 'Sell')

# Plotting
plt.plot(df['close'], label='Price')
plt.plot(df['short_ma'], label=f'Short MA ({short_window})')
plt.plot(df['long_ma'], label=f'Long MA ({long_window})')

# Plot signals
buy_signals = df[df['signals'] == 'Buy']
sell_signals = df[df['signals'] == 'Sell']
plt.scatter(buy_signals.index, buy_signals['close'], marker='^', color='g', label='Buy Signal')
plt.scatter(sell_signals.index, sell_signals['close'], marker='v', color='r', label='Sell Signal')

plt.title('Moving Average Crossover Strategy')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.show()

# Disconnect from MetaTrader 5
mt5.shutdown()
