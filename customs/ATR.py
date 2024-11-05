import MetaTrader5 as mt5
import pandas as pd
import numpy as np

if not mt5.initialize():
    print("initialize() failed, error code =",mt5.last_error())
    quit()


def get_atr_data(symbol, timeframe, start_pos=0):
    count = 100  # Number of bars to fetch (ample data for ATR calculation)
    atr_period = 14  # ATR period (standard is 14)
    # Fetch OHLC data from MetaTrader 5
    if start_pos == 0:
        atr_rates = mt5.copy_rates_from_pos(symbol, timeframe, start_pos, count)
    else:
        atr_rates = mt5.copy_rates_from(symbol, timeframe, start_pos, count)

    # Convert data into a pandas DataFrame for easier manipulation
    atr_df = pd.DataFrame(atr_rates)
    atr_df['time'] = pd.to_datetime(atr_df['time'], unit='s')  # Convert timestamps to human-readable format

    # --- Start ATR Calculation ---
    # 1. Calculate True Range (TR) components:
    atr_df['tr_high_low'] = atr_df['high'] - atr_df['low']  # Current High - Current Low
    atr_df['tr_high_prev_close'] = np.abs(atr_df['high'] - atr_df['close'].shift(1))  # High - Previous Close
    atr_df['tr_low_prev_close'] = np.abs(atr_df['low'] - atr_df['close'].shift(1))  # Low - Previous Close

    # 2. Calculate True Range (TR):
    atr_df['TR'] = atr_df[['tr_high_low', 'tr_high_prev_close', 'tr_low_prev_close']].max(axis=1)

    # 3. Calculate Average True Range (ATR) using Simple Moving Average (SMA):
    atr_df['ATR'] = atr_df['TR'].rolling(window=atr_period).mean()

    # --- ATR Percentage Calculation ---
    # ATR Percentage = (ATR / Close) * 100
    atr_df['ATR_Percentage'] = (atr_df['ATR'] / atr_df['close']) * 100

    # --- Difference Calculation ---
    # Calculate the difference between the current ATR and the previous row's ATR
    atr_df['ATR_Diff'] = atr_df['ATR'].diff()  # Difference of ATR values (positive or negative)
    atr_df['High_Diff'] = atr_df['high'].diff()  # Difference of ATR values (positive or negative)
    atr_df['Low_Diff'] = atr_df['low'].diff()  # Difference of ATR values (positive or negative)

    # Calculate the difference between the current ATR Percentage and the previous row's ATR Percentage
    atr_df['ATR_Percentage_Diff'] = atr_df['ATR_Percentage'].diff()  # Difference of ATR Percentage (positive or negative)

    # --- Sign Indicator ---
    # For actual ATR difference: +1 for positive change, -1 for negative change
    atr_df['ATR_Sign'] = np.sign(atr_df['ATR_Diff'])
    atr_df['High_Sign'] = np.sign(atr_df['High_Diff'])
    atr_df['Low_Sign'] = np.sign(atr_df['Low_Diff'])

    # For ATR percentage difference: +1 for positive change, -1 for negative change
    atr_df['ATR_Percentage_Sign'] = np.sign(atr_df['ATR_Percentage_Diff'])

    # For ATR percentage difference: +1 for positive change, -1 for negative change
    atr_df['ATR_Bool'] = np.where(atr_df['ATR_Percentage_Sign'] > 0, True, False)
    atr_df['High_Bool'] = np.where(atr_df['High_Sign'] > 0, True, False)
    atr_df['Low_Bool'] = np.where(atr_df['Low_Sign'] < 0, True, False)  # Low Bool is true if it's a lower low

    # Display the latest rows with ATR, ATR Percentage, and their differences and signs
    # print(atr_df[['time', 'ATR', 'ATR_Diff', 'ATR_Sign', 'ATR_Percentage', 'ATR_Percentage_Diff', 'ATR_Percentage_Sign', 'ATR_Percentage_Sign_Bool']].tail(10))
    # print(atr_df.iloc[-1]['ATR_Percentage'])
    return atr_df


if __name__ == '__main__':
    # Parameters
    symbol = "BTCUSDm"  # BTC/USD mini contract symbol
    timeframe = mt5.TIMEFRAME_M15 # 1-minute timeframe for detailed analysis
    start_pos = 0  # Start from the most recent bar
    print(get_atr_data(symbol, timeframe))