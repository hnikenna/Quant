import MetaTrader5 as mt5
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timezone
import pytz
# Initialize MT5 connection
if not mt5.initialize():
    print("Failed to initialize MT5")
    quit()


def make_trading_decision(symbol, utc_from=None, debug=True):
    import pandas as pd
    import numpy as np
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LinearRegression

    # GBP = 1, 2, 3, 5, 15
    # Timeframe is relative to stop loss size.
    # The smaller the sl size per trade, the smaller the timeframes for prediction
    timeframes = {
        '1 Min': mt5.TIMEFRAME_M1,
        # '1 Min2': mt5.TIMEFRAME_M1,
        # '1 Min3': mt5.TIMEFRAME_M1,
        # '2 Mins': mt5.TIMEFRAME_M2,
        # '3 Mins': mt5.TIMEFRAME_M3,
        # '4 Mins': mt5.TIMEFRAME_M4,
        # '5 Mins': mt5.TIMEFRAME_M5,
        # '10 Mins': mt5.TIMEFRAME_M10,
        # '15 Mins': mt5.TIMEFRAME_M15,
        # '30 Mins': mt5.TIMEFRAME_M30,
        # '1 Hr': mt5.TIMEFRAME_H1,
        # '2 Hr': mt5.TIMEFRAME_H2,
        # '3 Hr': mt5.TIMEFRAME_H3,
        # '4 Hr': mt5.TIMEFRAME_H4,
        # '6 Hr': mt5.TIMEFRAME_H6,
        # '12 Hr': mt5.TIMEFRAME_H12,

    }

    time_values = []
    rsi_values = []
    macd_values = []
    macd_signal_values = []
    ema_values = []
    price_differences = []

    scaling_factor_short = 3
    scaling_factor_long = 5
    for key, value in timeframes.items():
        time_values.append(key)
        # print(key, value)
        if utc_from is None:
            ticks = mt5.copy_rates_from_pos(symbol, value, 0, 21)
            history = mt5.copy_rates_from_pos(symbol, value, 0, 14)
            macd_ticks = mt5.copy_rates_from_pos(symbol, value, 0, 50)
            ema_ticks = mt5.copy_rates_from_pos(symbol, value, 0, 250)
        else:
            ticks = mt5.copy_rates_from(symbol, value, utc_from, 21)
            history = mt5.copy_rates_from(symbol, value, utc_from, 14)
            macd_ticks = mt5.copy_rates_from(symbol, value, utc_from, 50)
            ema_ticks = mt5.copy_rates_from(symbol, value, utc_from, 250)

            # print(ticks)

        hex_bytes = value.to_bytes(length=4, byteorder='little')

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

        rsi = rsi_df['rsi'].iloc[-1]
        rsi_values.append(rsi)
        # print('RSI:', rsi)

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
        # print(macd, signal)
        macd_values.append(macd)
        macd_signal_values.append(signal)
        # quit()

        # EMA
        # Convert data to DataFrame
        ema_df = pd.DataFrame(ema_ticks)

        # Calculate Exponential Moving Averages (EMA)
        # Determine short and long window scaling based on the timeframe
        # scaling_factor_short *= 2
        # scaling_factor_long *= 4
        scaling_factor_short = 9
        scaling_factor_long = 21

        # Calculate dynamic short and long windows
        short_window = min(int(scaling_factor_short), 50)
        long_window = min(int(scaling_factor_long), 200)

        ema_df['short_ema'] = ema_df['close'].ewm(span=short_window, adjust=False).mean()
        ema_df['long_ema'] = ema_df['close'].ewm(span=long_window, adjust=False).mean()

        # Generate buy/sell signals using loc to avoid the SettingWithCopyWarning
        ema_df.loc[short_window:, 'signal'] = 0  # 0 represents no action
        ema_df.loc[short_window:, 'signal'] = np.where(
            ema_df['short_ema'][short_window:] > ema_df['long_ema'][short_window:], 1, 0)  # Buy signal
        ema_df.loc[short_window:, 'signal'] = np.where(
            ema_df['short_ema'][short_window:] < ema_df['long_ema'][short_window:], -1,
            ema_df['signal'][short_window:])  # Sell signal

        # Print the DataFrame with signals
        # print(ema_df[['time', 'close', 'short_ema', 'long_ema', 'signal']])
        ema = ema_df['signal'].iloc[-1]
        # print('Last EMA:', ema)
        ema_values.append(ema)
        # quit()

        # create DataFrame out of the obtained data
        rates_frame = pd.DataFrame(ticks)

        # convert time in seconds into the datetime format
        rates_frame['time'] = pd.to_datetime(rates_frame['time'], unit='s')

        # display data
        # print("\nDisplay dataframe with data")
        # print(rates_frame)

        # Extract bid close prices
        close_prices = np.array([tick[1] for tick in ticks])

        # Prepare data for modeling
        X = close_prices[:-1].reshape(-1, 1)  # Feature: current close price
        y = close_prices[1:]  # Target: next close price

        # # print(y)
        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Create and train a Linear Regression model
        model = LinearRegression()
        model.fit(X_train, y_train)
        slope = model.coef_[0]
        # print('Slope:', slope)

        # Predict next close price
        latest_close_price = X[-1]
        # # print(np.array([[latest_close_price]]))
        next_predicted_close = model.predict(np.array([[latest_close_price]])[0])
        # print("Latest open price:", latest_close_price)
        # print("Last open price:", close_prices[-1])
        # print("Predicted next open price:", next_predicted_close)

        if utc_from is None:
            lasttick = mt5.symbol_info_tick(symbol)
            bid = lasttick.bid
        else:
            lasttick = mt5.copy_ticks_from(symbol, utc_from, 2, mt5.COPY_TICKS_ALL)[-1]
            bid = lasttick[1]
        # print(lasttick.bid, close_prices[-1], next_predicted_close)
        diff = (bid - float(next_predicted_close[0]))
        price_differences.append(diff)
        # print('Diff:', diff)

    normalized_differences = [pd / max(abs(pd) for pd in price_differences) if price_differences else 0 for pd in
                              price_differences]

    danger_zone_low = 20
    warning_zone_low = 30
    moderate_zone_low = 40
    moderate_zone_high = 100 - moderate_zone_low
    warning_zone_high = 100 - warning_zone_low
    danger_zone_high = 100 - danger_zone_low

    weight_very_strong = 4 / 4  # Adjust weights based on your strategy
    weight_strong = 3 / 4
    weight_moderate = 2 / 4
    weight_safe = 1 / 4

    buy_signals = 0
    sell_signals = 0

    macd_weight_very_strong = 3 / 3  # Adjust weights based on your strategy
    macd_weight_strong = 2 / 3
    macd_weight_moderate = 1 / 3

    macd_buy_signals = 0
    macd_sell_signals = 0

    ema_signal = 0

    action = ''
    signals = ''
    macd_signals = ''
    ema_signals = ''
    strength = sum(normalized_differences)

    for rsi, pd, tf in zip(rsi_values, normalized_differences, time_values):
        if debug:
            print(tf, end=' - ')
        if rsi < danger_zone_low and pd < 0:
            if debug:
                print(f"RSI: {rsi}, PD: {pd} - Buy (Danger Zone - Mega Buy Signal - Super Confident!)")
            buy_signals += weight_very_strong
            signals += 'B'
        elif danger_zone_low <= rsi < warning_zone_low and pd < 0:
            if debug:
                print(f"RSI: {rsi}, PD: {pd} - Buy (Warning Zone - Buy Signal Alert - Keep a Close Eye!)")
            buy_signals += weight_strong
            signals += 'B'
        elif warning_zone_low <= rsi < moderate_zone_low and pd < 0:
            if debug:
                print(f"RSI: {rsi}, PD: {pd} - Buy (Moderate Zone - Market on the Rise - Signal Developing!)")
            buy_signals += weight_moderate
            signals += 'B'
        elif warning_zone_low >= rsi and pd > 0:
            if debug:
                (f"RSI: {rsi}, PD: {pd} - Buy (Neutral Zone - Buy with Caution - Watch Out!)")
            buy_signals += weight_strong
            signals += 'B'
        elif moderate_zone_low <= rsi < moderate_zone_high:
            if pd > 0:
                if debug:
                    print(f"RSI: {rsi}, PD: {pd} - Buy (Safe Zone -  Market at Rest - Signal Not Clear!)")
                buy_signals += weight_safe
                signals += 'B'
            elif pd < 0:
                if debug:
                    print(f"RSI: {rsi}, PD: {pd} - Sell (Safe Zone - Market at Rest - Signal Not Clear!)")
                sell_signals += weight_safe
                signals += 'S'
        elif warning_zone_high <= rsi and pd < 0:
            if debug:
                print(f"RSI: {rsi}, PD: {pd} - Sell (Neutral Zone - Caution for Selling - Tread Carefully!)")
            sell_signals += weight_strong
            signals += 'S'
        elif moderate_zone_high <= rsi < warning_zone_high and pd > 0:
            if debug:
                print(f"RSI: {rsi}, PD: {pd} - Sell (Moderate Zone - Market on the Decline - Signal Developing!)")
            sell_signals += weight_moderate
            signals += 'S'
        elif warning_zone_high <= rsi < danger_zone_high and pd > 0:
            if debug:
                print(f"RSI: {rsi}, PD: {pd} - Sell (Warning Zone - Sell Signal Alert - Keep a Close Eye!)")
            sell_signals += weight_strong
            signals += 'S'
        elif danger_zone_high <= rsi and pd > 0:
            if debug:
                print(f"RSI: {rsi}, PD: {pd} - Sell (Danger Zone - Mega Sell Signal - Super Confident!)")
            sell_signals += weight_very_strong
            signals += 'S'
        else:
            if debug:
                print(f"RSI: {rsi}, PD: {pd} - Neutral (No Clear Signal)")
            signals += '-'

    for macd, signal, tf in zip(macd_values, macd_signal_values, time_values):
        if debug:
            print(tf, end=' - ')
        histogram = macd - signal

        if macd < 0 and signal < 0 and histogram > 0:
            if debug:
                print(f"MACD: {macd}, Signal: {signal}, Histogram: {histogram} - Buy (Strength increasing)")
            macd_buy_signals += macd_weight_very_strong
            macd_signals += 'B'
        elif macd > 0 and signal < 0 and histogram > 0:
            if debug:
                if debug:print(f"MACD: {macd}, Signal: {signal}, Histogram: {histogram} - Buy (Crossing to the upside)")
            macd_buy_signals += macd_weight_strong
            macd_signals += 'B'
        elif macd > 0 and signal > 0 and histogram > 0:
            if debug:print(f"MACD: {macd}, Signal: {signal}, Histogram: {histogram} - Buy (Strength decreasing)")
            macd_buy_signals += macd_weight_moderate
            macd_signals += 'B'
        elif macd > 0 and signal > 0 and histogram < 0:
            if debug:print(f"MACD: {macd}, Signal: {signal}, Histogram: {histogram} - Sell (Strength increasing)")
            macd_sell_signals += macd_weight_very_strong
            macd_signals += 'S'
        elif macd < 0 and signal > 0 and histogram < 0:
            if debug:print(f"MACD: {macd}, Signal: {signal}, Histogram: {histogram} - Sell (Crossing to the downside)")
            macd_sell_signals += macd_weight_strong
            macd_signals += 'S'
        elif macd < 0 and signal < 0 and histogram < 0:
            if debug:print(f"MACD: {macd}, Signal: {signal}, Histogram: {histogram} - Sell (Strength decreasing)")
            macd_sell_signals += macd_weight_moderate
            macd_signals += 'S'
        else:
            if debug:print(f"MACD: {macd}, Signal: {signal}, Histogram: {histogram} - Neutral (No clear trend)")
            macd_signals += '-'

    for val in ema_values:
        ema_signal += val
        if val > 0:
            ema_signals += 'B'
        elif val < 0:
            ema_signals += 'S'
        else:
            ema_signals += '-'

    # Compute Signals
    linear_strength = round(strength, 1)
    ema_strength = int(ema_signal)
    macd_strength = round(macd_buy_signals - macd_sell_signals, 1)
    rsi_strength = round(buy_signals - sell_signals, 1)

    final_signal = 0

    timeframe_count = len(timeframes)

    # We subtract one because we normalize the total from the max value
    # so it's almost never the exact amount. eg if we have [2, 4, 8] the total will be [0.25 + 0.5 + 1]
    max_value = timeframe_count - 1
    mid_value = round((timeframe_count / 2) + 0.01)
    excess_value = timeframe_count * 0.9  # 90% of max timeframes is almost guaranteed a sure signal

    rsi = rsi_strength
    l = linear_strength
    mcd = macd_strength

    mcd_relative_weight = timeframe_count - abs(mcd)
    if mcd < 0:
        mcd_relative_weight *= -1

    rsi_threshold = mid_value - max_value - abs(float(l))
    # rsi_threshold = min(excess_value, rsi_threshold)
    # if debug:print('Minimum RSI Threshold:', rsi_threshold)
    # if float(rsi) >= 0 and float(mcd) >= 0 and float(l) >= 0 and abs(float(rsi)) <= (timeframe_count / 2):
    #     final_signal = -1
    # elif float(rsi) <= 0 and float(mcd) <= 0 and float(l) <= 0 and abs(float(rsi)) <= (timeframe_count / 2):
    #     final_signal = 1
    # el
    if abs(float(rsi)) >= mid_value and abs(float(rsi)) >= rsi_threshold:
        # if debug:print('Following rsi...', rsi)
        final_signal = rsi
    else:
        if abs(float(macd_strength)) >= (timeframe_count / 2):
            # Inverse MACD Signal
            mcd = float(mcd) * -1
        final_signal = mcd
    # if debug:print('Final Signal', final_signal)
    if final_signal > 0:
        if debug:print("Buy (Positive Outlook)")
        action = 'buy'
    elif final_signal < 0:
        if debug:print("Sell (Negative Outlook)")
        action = 'sell'
    else:
        action = 'buy' if strength > 0 else 'sell'
        if debug:print(f"Neutral (No Clear Signal) - We'll Choose to {action.title()} from the Strength!")

    # action = 'buy' if rsi_strength > 0 else 'sell'
    back_tester_result = (rsi_strength * -1) + (macd_strength * -1) + (linear_strength * 0.5)

    if back_tester_result > 0:
        action = 'buy'
    elif back_tester_result <0:
        action = 'sell'
    else:
        action = None

    comment = 'R' + str(rsi_strength) + \
              'M' + str(macd_strength) + \
              'L' + str(linear_strength)
              # 'E' + str(ema_strength) + \

    output = {'action': action, 'strength': strength, 'signals': signals, 'ema_signals': ema_signals,
              'macd_signals': macd_signals, 'comment': comment}
    return output

# Your custom signal function
def signal_function(symbol, time):
    # print(time, type(time))
    # timezone = pytz.timezone("Etc/UTC")
    # time = datetime(2024, 7, 10, hour=4, minute=30, second=0, tzinfo=timezone)
    action = make_trading_decision(symbol=symbol, utc_from=time, debug=False)
    # print(action['action'])
    # quit()
    return action['action']

# Fetch historical data
def fetch_data(symbol, start_date, end_date, timeframe=mt5.TIMEFRAME_M5):
    rates = mt5.copy_rates_range(symbol, timeframe, start_date, end_date)
    if rates is None:
        print(f"Failed to get data for {symbol}")
        return None
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    return df

# Plot signals on the chart
def plot_signals(df, symbol):
    plt.figure(figsize=(12, 8))
    plt.plot(df['time'], df['close'], label=f'{symbol} Price', color='blue')

    buy_signals = []
    sell_signals = []

    for i, row in df.iterrows():
        # print(i, row['close'], df.iterrows()[0])
        signal = signal_function(symbol, row['time'].to_pydatetime().replace(tzinfo=timezone.utc))
        if signal == 'buy':
            buy_signals.append((row['time'], row['close']))
        elif signal == 'sell':
            sell_signals.append((row['time'], row['close']))

    if buy_signals:
        buy_times, buy_prices = zip(*buy_signals)
        plt.scatter(buy_times, buy_prices, marker='^', color='green', label='Buy Signal', s=100)

    if sell_signals:
        sell_times, sell_prices = zip(*sell_signals)
        plt.scatter(sell_times, sell_prices, marker='v', color='red', label='Sell Signal', s=100)

    plt.title(f'{symbol} Price with Buy/Sell Signals')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    plt.grid()
    plt.show()

# Example usage
symbol = 'EURUSDm'
start_date = datetime(2024, 8, 30, hour=9, minute=0, second=0, tzinfo=timezone.utc)
end_date = datetime(2024, 8, 30, hour=12, minute=59, second=59, tzinfo=timezone.utc)

df = fetch_data(symbol, start_date, end_date)
if df is not None:
    plot_signals(df, symbol)

# Shutdown MT5 connection
mt5.shutdown()
