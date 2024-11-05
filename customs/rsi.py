import MetaTrader5 as mt5
import pandas as pd
import matplotlib.pyplot as plt

# establish connection to MetaTrader 5 terminal
if not mt5.initialize():
    print("initialize() failed, error code =",mt5.last_error())
    quit()
# Specify the symbol and timeframe
symbol = "BTCUSDm"  # Replace with the desired financial instrument
timeframe = mt5.TIMEFRAME_M15  # Replace with the desired timeframe

# Request historical data
history = mt5.copy_rates_from_pos(symbol, timeframe, 0, 14)
# Create a Pandas DataFrame from the historical data
rsi_df = pd.DataFrame(history)
rsi_df['time'] = pd.to_datetime(rsi_df['time'], unit='s')
rsi_df.set_index('time', inplace=True)

# Calculate RSI
period = 14
rsi_df['delta'] = rsi_df['close'].diff()
rsi_df['gain'] = rsi_df['delta'].apply(lambda x: x if x > 0 else 0)
rsi_df['loss'] = rsi_df['delta'].apply(lambda x: -x if x < 0 else 0)
average_gain = rsi_df['gain'].rolling(window=period).mean()
average_loss = rsi_df['loss'].rolling(window=period).mean()
rs = average_gain / average_loss
rsi_df['rsi'] = 100 - (100 / (1 + rs))

print(rsi_df['rsi'][-1])
# Plot the RSI
plt.figure(figsize=(12, 6))
plt.plot(rsi_df.index, rsi_df['rsi'], label='RSI', color='blue')
plt.title(f'{symbol} RSI ({period} periods)')
plt.xlabel('Time')
plt.ylabel('RSI Value')
plt.legend()
# plt.show()
