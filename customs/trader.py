import random, math
import MetaTrader5 as mt5
from datetime import datetime, timedelta, time
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
h1 = ("BTCUSDm", 7500, 40, 0.01, 3, 0.1)
h2 = ("US30m", 500, 7, 0.02, 3, 0.1)
# h2 = ("US30m", 750, 7, 0.01, 3, 0.1)
h3 = ("ETHUSDm", 750, 4, 0.1, 3, 1)
h4 = ("XAUUSDm", 750,  0.2, 0.01, 3, 1)
h5 = ("EURUSDm", 75, 0, 0.01, 3, 0.1)      # Three trades instantly hit stop loss without the actual price hitting stop loss. trade this symbol at your own risk
h6 = ("GBPUSDm", 75, 0, 0.01, 3, 10000)      # Three trades instantly hit stop loss without the actual price hitting stop loss. trade this symbol at your own risk
h7 = ("EURJPYm", 75, 0.02, 0.01, 3, 0.1)
h8 = ("USDCHFm", 75, 0, 0.01, 3, 0.1)
h9 = ("AUDUSDm", 75, 0.0, 0.01, 3, 0.1)
h10 = ("USDJPYm", 75, 0.02, 0.01, 3, 0.1)

headers = [h10]
header = random.choice(headers)

timer = 30
incr = 0
max_positions = 1
long_timer = timer * max_positions
magic = random.randint(1000, 9999)
# comment = str(random.randint(100000, 999999))
random_action = ''
# random_action_selector = pyip.inputChoice(choices=['buy', 'sell'], blank=True)
process = None

id_checker = False

is_trending = False


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
        # condition = mt5.positions_total()
        condition = False
        if condition:
            # If the condition is true and the script is not running, start the script
            if process is None or process.poll() is not None:
                print("Buy and Sell Noticed! Starting the script.")
                process = subprocess.Popen(["python", "scalper.py"])
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
        # print(duration)
        totalpos = mt5.positions_total()
        if duration >= timer:
            break
            # return
        elif totalpos < 1:
            break
    return None


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
        mean_price = sum(close_prices)/len(close_prices)
        print('Mean Price:', mean_price)

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

        print(lasttick.bid)
        print('Deviation:', (((lasttick.bid / mean_price)*100)-100))

        if float(next_predicted_close[0]) < lasttick.bid:
            actions.append('buy')
        else:
            actions.append('sell')

    return actions


last_ticket = {'id': None}


def get_last_ticket():

    global last_ticket, id_checker, is_trending
    # get the number of deals in history

    count_limit = 60

    count = 0
    today = datetime.now().date()
    # Calculate yesterday's date
    yesterday = today - timedelta(days=1)
    from_date = datetime.combine(yesterday, time())

    # get deals for symbols
    while True:
    # while count <= count_limit:
    #     count += 1

        to_date = datetime.now()
        deals = mt5.history_deals_get(from_date, to_date)
        # print(deals[-1])
        the_deal = None

        # print('Last Deal:', deals[-1])
        # print('Last Deal2:', deals[-2])
        # print('Last Deal3:', deals[-3])
        deals = deals[::-1]
        if deals == None:
            print("No deals, error code={}".format(mt5.last_error()))
        elif len(deals) > 0:
            for deal in deals:
                try:
                    deal_point = mt5.symbol_info(deal.symbol).point
                    tmp_incr = math.log((deal.volume * 1/deal_point), 2)
                    # print('Temp Increment:', tmp_incr)
                except:
                    continue
                if deal.reason == 3:
                    # print(f'Error with position {deal.position_id} reason')
                    continue
                elif deal.reason == 4:
                    # Loss
                    print(f'You made a loss of {deal.profit}')
                    the_deal = {'id': deal.position_id, 'profit': False}


                    # print(deal)
                    break
                elif deal.reason == 5:
                    # Profit
                    print(f'You made a profit of {deal.profit}')
                    the_deal = {'id': deal.position_id, 'profit': True}
                    # print(deal)
                    break
                else:
                    # print(f'Error with position {deal.position_id} reason')
                    continue
                # print("  ",deal)

            if id_checker:
                if str(id_checker) != str(the_deal['id']):
                    # pass
                    id_checker = False
                    pass
                else:
                    print(f'Haven\'t seen the latest position. retrying...')
                    continue

            if 'id' in the_deal and the_deal['id'] != last_ticket['id']:

                # if 'profit' in last_ticket and not last_ticket['profit'] and not the_deal['profit']:
                #     is_trending = not is_trending
                last_ticket = the_deal
                the_deal['incr'] = tmp_incr + 1
                if tmp_incr > 2:
                    the_deal['incr'] = 0
                print('Deal:', the_deal)
                return the_deal
            else:

                print(f'Deal error - "{the_deal}" - retrying...')


if mt5.positions_total() >= max_positions:
    checker = get_last_ticket()
    id_checker = checker['id']
    if checker['profit'] == False:
        is_trending = not is_trending

# get_last_ticket()

last_header = None

while True:


    header = random.choice(headers)
    timer_start_time = datetime.now()

    # if header == last_header:
    #     continue

    symbol = header[0]
    mt5.symbol_select(symbol,True)
    # ticker = (mt5.symbol_info_tick("ETHUSDm"))
    # print(ticker)
    spread = round(float(mt5.symbol_info_tick(symbol).ask) - float(mt5.symbol_info_tick(symbol).bid), 3)
    point = mt5.symbol_info(symbol).point
    # print('Point:', point)
    sl= (header[1])
    tp = (sl + (spread*(1/point))) * 5
    spread_limit = header[2]

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
        # '1 Mins': mt5.TIMEFRAME_M1,
        # '2 Mins': mt5.TIMEFRAME_M2,
        # '3 Mins': mt5.TIMEFRAME_M3,
        # '4 Mins': mt5.TIMEFRAME_M4,
        # '5 Mins': mt5.TIMEFRAME_M5,
        # '10 Mins': mt5.TIMEFRAME_M10,
        '15 Mins': mt5.TIMEFRAME_M15,
        # '30 Mins': mt5.TIMEFRAME_M30,
        # '1 Hr': mt5.TIMEFRAME_H1,
        # '2 Hr': mt5.TIMEFRAME_H2,
        # '4 Hr': mt5.TIMEFRAME_H4,
    }
    # get_actions(timeframes)
    # quit()
    buy_total = 0
    sell_total = 0

    for key, value in timeframes.items():
        # print(key, value)
        ticks = mt5.copy_rates_from_pos(symbol, value, 0, 21)
        history = mt5.copy_rates_from_pos(symbol, value, 0, 14)

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

        rsi = rsi_df['rsi'][-1]
        print('RSI:', rsi)
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
        print('Slope:', slope)

        # Predict next close price
        latest_close_price = X[-1]
        # # print(np.array([[latest_close_price]]))
        next_predicted_close = model.predict(np.array([[latest_close_price]])[0])
        # print("Latest open price:", latest_close_price)
        # print("Last open price:", close_prices[-1])
        # print("Predicted next open price:", next_predicted_close)

        lasttick=mt5.symbol_info_tick(symbol)
        # print(lasttick.bid, close_prices[-1], next_predicted_close)
        diff = (lasttick.bid - float(next_predicted_close[0]))
        print('Diff:', diff)
        quit()


        if float(next_predicted_close[0]) < lasttick.bid:
            val = round(lasttick.bid - float(next_predicted_close[0]), 8)
            print(f'{key} - Buy {symbol} '
                  f'by {val} points '
                  f'than the previous price ({float(next_predicted_close[0])}).')
            actions.append('buy')
            buy_total += val
            comment += 'B'

            # print('Buy Total:', buy_total)
        else:
            val = round(float(next_predicted_close[0]) - lasttick.bid, 8)
            print(f'{key} - Sell {symbol} '
                  f'by {val} points '
                  f'than the previous price ({float(next_predicted_close[0])}).')
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
    print(f'\nBased on prediction we should be {action}ing {symbol} by {full_strength}% - #{comment} - RSI {rsi}')
    # random_action = pyip.inputChoice(choices=['buy', 'sell'], blank=True)
    # print(f'\nBut instead we will be {random_action}ing!')
    #
    # action = random_action

    point = mt5.symbol_info(symbol).point
    price = mt5.symbol_info_tick(symbol).ask
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

    last_pos = get_last_ticket()
    # print(last_pos)
    if last_pos:
        if last_pos['profit']:
            incr = 0
        else:
            incr = last_pos['incr']

            # Change the trending status
            is_trending = not is_trending

    sqr = (2**incr)
    # print('Square Value:', sqr)
    lot = header[3] * sqr

    print(f'Is Trend Detected? {is_trending}')
    if not is_trending:
        if action == 'buy':
            action = 'sell'
        else:
            action = 'buy'

    if action == 'buy':

        # new_sl = tp
        # tp = sl
        # sl = new_sl

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

    timer_end_time = datetime.now()

    # Calculate the duration in seconds
    duration = (timer_end_time - timer_start_time).total_seconds()

    if duration > 5:
        print(f'Duration greater than 5secs - {duration}secs - Retrying...')
        # Reset the ticket id duplicate checker
        last_ticket['id'] = None
        continue
        # if not last_pos['profit']:



    # send a trading request
    result = mt5.order_send(request)
    if result and result.retcode != mt5.TRADE_RETCODE_DONE:
        result_dict = result._asdict()
        print(result.comment)
        last_ticket['id'] = None
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
