import MetaTrader5 as mt5
import pandas as pd

# Connect to MetaTrader 5
mt5.initialize()

# Specify the symbol and timeframe
symbol = "BTCUSDm"
timeframe = mt5.TIMEFRAME_M15

# Request historical data
history = mt5.copy_rates_from_pos(symbol, timeframe, 0, 2100)

# Create a DataFrame from the historical data
if len(history) < 1:
    print('history error!')
    quit()
df = pd.DataFrame(history)
df['time'] = pd.to_datetime(df['time'], unit='s')
df.set_index('time', inplace=True)

# Calculate the Average True Range (ATR)
df['ATR'] = df['high'] - df['low']

# Check if the market is trending
average_atr = df['ATR'].mean()
current_atr = df['ATR'].iloc[-1]

if current_atr > 1.5 * average_atr:
    print("The market is in a trending state.")
else:
    print("The market is not trending significantly.")

# Disconnect from MetaTrader 5
mt5.shutdown()
