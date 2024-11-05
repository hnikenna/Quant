print('Initializing...')
import random
import MetaTrader5 as mt5
from datetime import datetime
import numpy as np
import subprocess
from time import sleep
from datetime import datetime
import pandas as pd
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# for i in range(7):
#     print(random.randint(10, 25))

import random

# print(random.randint(10, 25))


# Initialize MetaTrader 5
if not mt5.initialize():
    print("initialize() failed, error code =", mt5.last_error())
    quit()

# (SYMBOL, STOP_LOSS, SPREAD_LIMIT, VOLUME, PROFIT, STRENGTH)
# header = ("US30m", 750, 7, 0.01, 3, 0.1)
# header = ("EURUSDm", 75, 0, 0.01, 3, 0.1)
# header = ("USDCHFm", 75, 0, 0.01, 3, 0.1)
header = ("UKOILm", 750, 0.12, 0.01, 3, 0.1)
# header = ("AUDUSDm", 75, 0.0, 0.01, 3, 0.1)
# header = ("USDJPYm", 75, 0.02, 0.01, 3, 0.1)
# header = ("EURJPYm", 75, 0.02, 0.01, 3, 0.1)

symbol = header[0]
sl= header[1]
tp = (sl * header[4])
spread_limit = header[2]
lot = header[3]

timer = 250
max_positions = 30
long_timer = timer * max_positions
magic = random.randint(1000, 9999)
# comment = str(random.randint(100000, 999999))
random_action = ''
# random_action_selector = pyip.inputChoice(choices=['buy', 'sell'], blank=True)
process = None


def scleep(timer:int):
    global actions, process, timeframes

    start_time = datetime.now()

    while True:
        # new_actions = get_actions(timeframes)
        # # print(new_actions)
        #
        # # Scalp if irregularity is detected
        # checklist = [1 if act == 'buy' else 0 for act in new_actions]
        # condition = bool(1 in checklist and 0 in checklist)
        # condition = False
        condition = mt5.positions_total()
        if condition:
            # If the condition is true and the script is not running, start the script
            if process is None or process.poll() is not None:
                print("Buy and Sell Noticed! Starting the script.")
                process = subprocess.Popen(["python", "scalper-mini.py"])
        else:
            # If the condition becomes false and the script is running, terminate it
            # subprocess.Popen(["python", "scalper.py"]).terminate()
            if process is not None and process.poll() is None:
                print("No buy and Sell notice! Terminating the script.")
                process.terminate()
        # End time
        end_time = datetime.now()

        # Calculate the duration in seconds
        duration = (end_time - start_time).total_seconds()

        if duration >= timer:
            return
        elif mt5.positions_total() < 1:
            return 


def get_actions(timeframes:dict):
    actions = []

    for key, value in timeframes.items():
        # print(key, value)
        ticks = mt5.copy_rates_from_pos(symbol, value, 0, 21)

        # create DataFrame out of the obtained data
        rates_frame = pd.DataFrame(ticks)

        # convert time in seconds into the datetime format
        rates_frame['time'] = pd.to_datetime(rates_frame['time'], unit='s')


        # Extract bid close prices
        close_prices = np.array([tick[1] for tick in ticks])

        # Prepare data for modeling
        X = close_prices[:-1].reshape(-1, 1)  # Feature: current close price
        y = close_prices[1:]  # Target: next close price

        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Create and train a Linear Regression model
        model = LinearRegression()
        model.fit(X_train, y_train)

        # Predict next close price
        latest_close_price = X[-1]
        next_predicted_close = model.predict(np.array([[latest_close_price]])[0])

        lasttick=mt5.symbol_info_tick(symbol)

        if float(next_predicted_close[0]) < lasttick.bid:
            actions.append('buy')
        else:
            actions.append('sell')

    return actions

last_header = None
while True:


    # if random_action_selector == '':
    #     random_action = random.choice(['buy', 'sell'])
    # else:
    #     random_action = random_action_selector
    # if random_action == 'buy':
    #     random_action = 'sell'
    # else:
    #     random_action = 'buy'

    comment = ''
    # comment = str(random.randint(100000, 999999))
    actions = []
    checklist = []

    # GBP = 1, 2, 3, 5, 15

    timeframes = {
        '1 Mins': mt5.TIMEFRAME_M1,
        '2 Mins': mt5.TIMEFRAME_M2,
        '3 Mins': mt5.TIMEFRAME_M3,
        '4 Mins': mt5.TIMEFRAME_M4,
        '5 Mins': mt5.TIMEFRAME_M5,
        # '10 Mins': mt5.TIMEFRAME_M10,
        # '15 Mins': mt5.TIMEFRAME_M15,
        # '30 Mins': mt5.TIMEFRAME_M30,
        # '1 Hr': mt5.TIMEFRAME_H1,
        # '2 Hr': mt5.TIMEFRAME_H2,
        # '4 Hr': mt5.TIMEFRAME_H4,
    }

    buy_total = 0
    sell_total = 0

    for key, value in timeframes.items():
        # print(key, value)
        ticks = mt5.copy_rates_from_pos(symbol, value, 0, 21)

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

        # Predict next close price
        latest_close_price = X[-1]
        # # print(np.array([[latest_close_price]]))
        next_predicted_close = model.predict(np.array([[latest_close_price]])[0])
        # print("Latest open price:", latest_close_price)
        # print("Last open price:", close_prices[-1])
        # print("Predicted next open price:", next_predicted_close)

        lasttick=mt5.symbol_info_tick(symbol)
        # print(lasttick.bid, close_prices[-1], next_predicted_close)

        if float(next_predicted_close[0]) < lasttick.bid:
            val = round(lasttick.bid - float(next_predicted_close[0]), 8)
            print(f'{key} - Buy {symbol} '
                  f'by {val} points '
                  f'than the previous open price.')
            actions.append('buy')
            buy_total += val
            comment += 'B'

            # print('Buy Total:', buy_total)
        else:
            val = round(float(next_predicted_close[0]) - lasttick.bid, 8)
            print(f'{key} - Sell {symbol} '
                  f'by {val} points '
                  f'than the previous open price.')
            actions.append('sell')
            sell_total += val
            comment += 'S'
            # print('Sell Total:', sell_total)
        t_total = 0
        f_total = 0
        if random_action:
            action = random_action
        elif sell_total < buy_total:
            action = 'buy'
            t_total = buy_total - sell_total
            f_total = buy_total
        elif sell_total > buy_total:
            action = 'sell'
            t_total = sell_total - buy_total
            f_total = sell_total
        else:
            action = max(actions, key=Counter(actions).get)
        # action = max(actions, key=Counter(actions).get)
    strength = round(float(t_total)*header[5], 2)
    full_strength = round(float(f_total)*header[5], 2)
    comment += f'- {strength}%'
    print(f'\nBased on prediction we should be {action}ing {symbol} by {full_strength}% - #{comment}')
    # random_action = pyip.inputChoice(choices=['buy', 'sell'], blank=True)
    # print(f'\nBut instead we will be {random_action}ing!')
    #
    # action = random_action

    point = mt5.symbol_info(symbol).point
    price = mt5.symbol_info_tick(symbol).ask
    spread = round(float(mt5.symbol_info_tick(symbol).ask) - float(mt5.symbol_info_tick(symbol).bid), 3)
    positions_total=mt5.positions_total()


    print('Spread:', spread)
    if spread > spread_limit:
        print(f'Spread too large. Auto retry in {timer}s...')
        scleep(timer)
        continue
    # print(positions_total)
    if positions_total >= max_positions:
        print(f"You have a {positions_total} open positions limit. We can't open any more. \nAuto retry in {long_timer}s...")
        scleep(long_timer)
        continue

    deviation = 20

    if action == 'buy':
        order_type = mt5.ORDER_TYPE_BUY
        order_sl = price - sl * point
        order_tp = price + tp * point
    else:
        order_type = mt5.ORDER_TYPE_SELL
        order_sl = price + sl * point
        order_tp = price - tp * point

    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": lot,
        "type": order_type,
        "price": price,
        "sl": order_sl,
        "tp": order_tp,
        "deviation": deviation,
        "magic": magic,
        "comment": comment,
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_FOK,
    }

    # send a trading request
    result = mt5.order_send(request)
    if result and result.retcode != mt5.TRADE_RETCODE_DONE:
        result_dict = result._asdict()
        print(result.comment)

        print(f'\nRetrying in {timer}s...\n')
        scleep(timer)
        continue
    elif not result:
        print('Error sending request!')
        input('Enter any value to continue | ')
        continue
    # else:
    #     print(result)

    # print("2. order_send done, ", result)
    print("   opened position with POSITION_TICKET={}".format(result.order))
    last_header = header

    # timer = 30
    print(f"\nRetrading in {timer}s...")
    # input('|  ')
    scleep(timer)
# Shutdown MetaTrader 5
mt5.shutdown()
