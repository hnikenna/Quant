import random
import MetaTrader5 as mt5
from datetime import datetime, timedelta, time, timezone
from datetime import datetime
import pandas as pd
import numpy as np


def calculate_linear_scaling_percentage(profit, max_profit):
    # max_profit = 3
    weight = profit / max_profit  # Normalize the profit to a scale of 0 to 1

    scale = 25
    opp = 100 - scale
    # Map the normalized weight to the percentage scale between 25% and 75%
    scaling_percentage = scale + (opp - scale) * weight

    scaler = scaling_percentage
    # scaler = min(max(scaling_percentage, scale), opp)
    return profit * (scaler / 100)


def dynamic_tp_adjustment(current_price, scaled_tp, tp_adjust_threshold=50, tp_increase_amount=2500):
    # print('Take Profit Threshold for Current Profit:', scaled_tp * (1 - tp_adjust_threshold / 100))
    # Dynamic TP adjustment: If the current price is within tp_adjust_threshold of the TP, increase TP
    if True or (current_price >= scaled_tp * (1 - tp_adjust_threshold / 100)):
        scaled_tp = current_price + (tp_increase_amount / 100)
    return scaled_tp


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
    atr_df['High_Diff'] = atr_df['high'].diff() * -1
    atr_df['Low_Diff'] = atr_df['low'].diff()  # Difference of ATR values (positive or negative)

    # Calculate the difference between the current ATR Percentage and the previous row's ATR Percentage
    atr_df['ATR_Percentage_Diff'] = atr_df[
        'ATR_Percentage'].diff()  # Difference of ATR Percentage (positive or negative)

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
    atr_df['Low_Bool'] = np.where(atr_df['Low_Sign'] > 0, True, False)  # Low Bool is true if it's a lower low

    # Display the latest rows with ATR, ATR Percentage, and their differences and signs
    # print(atr_df[['time', 'ATR', 'ATR_Diff', 'ATR_Sign', 'ATR_Percentage', 'ATR_Percentage_Diff', 'ATR_Percentage_Sign', 'ATR_Percentage_Sign_Bool']].tail(10))
    # print(atr_df.iloc[-1]['ATR_Percentage'])
    return atr_df


scalp_profit = 0.05
scalp_max_loss = 0.02
trail_profit = 8
trail_change = 2
scalp_profit_dict = {}
scalp_max_profit_dict = {}
scalp_symbols, trail_symbols = ([], [])
bypass_risk = False  # Turn off in production!
force_scalp = False  # Turn off in production!
account_balance_starter = 20
super_trending = False


def trade_management():
    global scalp_profit_dict, scalp_max_profit_dict, scalp_symbols, trail_symbols, \
        scalp_profit, trail_change, trail_profit, bypass_risk, scalp_max_loss, force_scalp, \
        account_balance_starter, super_trending

    timeframes = {
        # 1440: mt5.TIMEFRAME_D1,
        # 720: mt5.TIMEFRAME_H12,
        # 480: mt5.TIMEFRAME_H8,
        # 360: mt5.TIMEFRAME_H6,
        # 240: mt5.TIMEFRAME_H4,
        # 180: mt5.TIMEFRAME_H3,
        # 120: mt5.TIMEFRAME_H2,
        # 60: mt5.TIMEFRAME_H1,
        30: mt5.TIMEFRAME_M30,
        # 20: mt5.TIMEFRAME_M20,
        15: mt5.TIMEFRAME_M15,
        # 12: mt5.TIMEFRAME_M12,
        # 10: mt5.TIMEFRAME_M10,
        # 6: mt5.TIMEFRAME_M6,
        # 5: mt5.TIMEFRAME_M5,
        # 4: mt5.TIMEFRAME_M4,
        # 3: mt5.TIMEFRAME_M3,
        # 2: mt5.TIMEFRAME_M2,
        # 1: mt5.TIMEFRAME_M1
    }

    positions = mt5.positions_get()
    account_info = mt5.account_info()

    balance = account_info.balance
    equity = account_info.equity
    account_profit = account_info.profit

    if not super_trending and ((balance * 2.1) <= equity) and (equity >= account_balance_starter):
        force_scalp = True
    elif force_scalp and (account_profit <= 0):
        print('Turning off Force Scalping!')
        force_scalp = False

    # print(scalp_symbols, trail_symbols)

    if len(positions) > 0:
        for position in positions:

            has_two_candlesticks_passed = False
            for time_int, mt5_time in timeframes.items():
                market_trending = get_atr_data(position.symbol, mt5_time).iloc[-2]
                # Latest candlestick values are dynamic. and will most times not open at a higher high or lower low,
                # so we use the previous candlestick_before it to check hence iloc[-2]

                order_type = 'buy' if position.type == 0 else 'sell'
                ###

                # Define the duration of one candlestick (15 minutes)
                candlestick_duration = timedelta(minutes=time_int)

                # Define the duration for two candlesticks
                required_duration = 1 * candlestick_duration

                # Function to round down to the nearest 15-minute interval
                def round_down_to_interval(dt, interval):
                    # Ensure the datetime is in UTC
                    if dt.tzinfo is None:
                        dt = dt.replace(tzinfo=timezone.utc)  # Make naive datetime aware if necessary
                    # Calculate total seconds since the start of the day
                    total_seconds = (dt - dt.replace(hour=0, minute=0, second=0, microsecond=0)).total_seconds()
                    interval_seconds = interval.total_seconds()
                    # Calculate the number of complete intervals
                    rounded_seconds = (total_seconds // interval_seconds) * interval_seconds
                    # Compute the new rounded datetime
                    rounded_time = dt.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(
                        seconds=rounded_seconds)
                    return rounded_time

                # Retrieve and convert timestamps
                trade_open_timestamp = position.time_update  # Unix timestamp when the trade was opened
                trade_open_time = datetime.fromtimestamp(trade_open_timestamp,
                                                         tz=timezone.utc)  # Convert to UTC datetime object

                # Current candle time (in the format 'YYYY-MM-DD HH:MM:SS')
                candle_time_str = str(market_trending['time'])
                candle_time = datetime.strptime(candle_time_str, '%Y-%m-%d %H:%M:%S').replace(
                    tzinfo=timezone.utc)  # Convert to UTC datetime object

                # Round trade open time to the nearest candlestick interval
                rounded_trade_open_time = round_down_to_interval(trade_open_time, candlestick_duration)

                # Check if at least two candlesticks have passed
                has_two_candlesticks_passed = (candle_time >= rounded_trade_open_time) and (
                            (candle_time - rounded_trade_open_time) >= required_duration)

                if has_two_candlesticks_passed is True:
                    break

                # # Print result for debugging
                # print(f"Trade open time (UTC): {trade_open_time}")
                # print(f"Rounded trade open time (UTC): {rounded_trade_open_time}")
                # print(f"Current candle time (UTC): {candle_time}")
                # print(f"Time difference: {candle_time - rounded_trade_open_time}")
                # print(f"Required duration: {required_duration}")
                # print(f"Has two candlesticks passed? {has_two_candlesticks_passed}")
                # quit()
                ###

            if order_type == 'buy':
                is_market_not_trending = market_trending['Low_Bool'] and (
                    market_trending['Low_Diff'] > market_trending['High_Diff'])
            elif order_type == 'sell':
                is_market_not_trending = market_trending['High_Bool'] and (
                    market_trending['High_Diff'] > market_trending['Low_Diff'])

            # SCALP
            if has_two_candlesticks_passed and not is_market_not_trending:
                # if position.symbol in scalp_symbols:

                if scalp_profit < scalp_max_loss and not bypass_risk:
                    print(f'Profit too low tf? Enter any value to proceed with a profit of {scalp_profit}.')
                    input()

                # print(scalp_profit_dict)
                # print(f'Scanning for {scalp_profit} profit...')

                scalp_profit_dict[position.ticket] = position.profit
                # print(max_profit_dict)
                try:
                    scalp_max_profit = scalp_max_profit_dict[position.ticket]
                except:
                    scalp_max_profit = scalp_max_loss

                # if position.profit > 0:
                #     print(position.profit)
                # print(position)
                # print(scalp_max_profit, position.profit, scalp_profit)
                if scalp_max_profit < position.profit > scalp_profit:
                    # Update Max Profit for This Position
                    scalp_max_profit = position.profit
                    scalp_max_profit_dict[position.ticket] = position.profit
                    # print(f'{position.symbol} max profit {scalp_max_profit}')

                # print(max_profit)
                # print(scalp_profit, position.profit, scalp_max_profit, scalp_max_profit_dict)
                if scalp_profit < position.profit < scalp_max_profit:
                    print('-profit hehe!')

                    order_type = mt5.ORDER_TYPE_SELL if position.type == 0 else mt5.ORDER_TYPE_BUY

                    trade_request = {
                        "action": mt5.TRADE_ACTION_DEAL,
                        "symbol": position.symbol,
                        "volume": position.volume,  # Specify the volume to close
                        "type": order_type,  # Specify the order type (SELL to close a long position)
                        "position": position.ticket,  # Ticket number of the position to close
                    }

                    # Send the trade request to close the position
                    result = mt5.order_send(trade_request)

                    if result.retcode == mt5.TRADE_RETCODE_DONE:

                        scalp_max_profit = scalp_max_loss
                        scalp_max_profit_dict[position.ticket] = scalp_max_loss
                        del scalp_max_profit_dict[position.ticket]
                        del scalp_profit_dict[position.ticket]
                        print(f"Position closed successfully.")
                        # print(f'\nScanning for {scalp_profit} profit...')

                    else:
                        print(f"Failed to close position. {trade_request} - Error:", result.comment)

            # TRAIL
            if True:
                # elif position.symbol in trail_symbols:

                # # Remove positions data from scalping db for efficiency
                # try:
                #     del scalp_max_profit_dict[position.ticket]
                # except:
                #     pass
                # try:
                #     del scalp_profit_dict[position.ticket]
                # except:
                #     pass

                position_percentage_change = round(
                    ((abs(position.price_open - position.price_current) / position.price_open) * 100), 2)

                if position.profit < 0:
                    position_percentage_change *= -1

                # print(f'{position.symbol} Percentage Change: {position_percentage_change}%')

                if trail_profit < position.profit:
                    # if trail_change < position_percentage_change:

                    pip_scale = float(abs((position.price_open - position.price_current) / position.profit))
                    # print('Pip Scale:', pip_scale)

                    if order_type == 'sell':
                        current_profit = float((position.price_open - position.price_current) / pip_scale)
                    else:
                        current_profit = float((position.price_current - position.price_open) / pip_scale)

                    # print('Current Profit:', current_profit)
                    # print(current_profit == position.profit)

                    get_scaled_tp = float(abs((position.price_open - position.tp) / pip_scale))
                    # print('TP:',get_scaled_tp)

                    # print(get_scaled_tp)
                    # print('Scaled Tp:', scaled_tp)

                    # Increase TP if current profit is close
                    tolerance = 0.2  # the lower the tolerance the closer it needs to be to the tp
                    # Calculate the tolerance range based on the target profit
                    tolerance_range = get_scaled_tp * tolerance

                    # Check if the current profit is within the tolerance range of the target profit
                    is_profit_too_close = (position.profit / get_scaled_tp) >= tolerance

                    # if 'xau' in str(position.symbol).lower():
                    #     print((position.profit / get_scaled_tp), position.symbol)

                    # is_profit_too_close = abs(position.profit - get_scaled_tp) <= tolerance_range

                    # print(tolerance_range, abs(position.profit - get_scaled_tp))
                    if not is_profit_too_close:
                        continue
                    else:
                        scaled_tp = get_scaled_tp * 8 * tolerance

                    # print('*', end='')

                    # Dynamically adjust the TP

                    new_tp = position.tp

                    # Update tp if scaled one is larger
                    # print(scaled_tp > get_scaled_tp)
                    if scaled_tp > get_scaled_tp:
                        # print('upgrading tp')
                        tp_increment = scaled_tp * pip_scale

                        new_tp = (position.price_open + tp_increment) if order_type == 'buy' else (
                                position.price_open - tp_increment)

                    if order_type == 'sell':
                        get_scaled_sl = (position.price_open - position.sl) / pip_scale
                    else:
                        get_scaled_sl = (position.sl - position.price_open) / pip_scale
                    # print(position.price_open, position.sl, pip_scale)
                    # print(position.price_open - position.sl)
                    # print((position.price_open - position.sl) / pip_scale)
                    # print(get_scaled_sl)
                    # print('SL', get_scaled_sl)

                    scaled_sl = calculate_linear_scaling_percentage(position.profit, scaled_tp)
                    # print('Scaled Sl:', scaled_sl)
                    new_sl = position.sl

                    # # Update sl if scaled one is larger
                    # if scaled_sl > get_scaled_sl:
                    #     # print('upgrading sl')
                    #     sl_increment = scaled_sl * pip_scale
                    #
                    #     new_sl = (position.price_open + sl_increment) if order_type == 'buy' else (
                    #                 position.price_open - sl_increment)
                    # quit()
                    # print('tst')
                    if new_tp == position.tp and new_sl == position.sl:
                        # print('No changes..')
                        # time.sleep(timer)
                        continue
                    # print(new_tp, position.tp, new_sl, position.sl)
                    trade_request = {
                        "action": mt5.TRADE_ACTION_SLTP,
                        "symbol": position.symbol,
                        # "volume": position.volume,  # Specify the volume to close
                        # "type": order_type,  # Specify the order type (SELL to close a long position)
                        "position": position.ticket,  # Ticket number of the position to close
                        'sl': new_sl,
                        'tp': new_tp,

                    }

                    # Send the trade request to close the position
                    result = mt5.order_send(trade_request)

                    if result.retcode == mt5.TRADE_RETCODE_DONE:
                        print(f"Trailed position {position.ticket} successfully.")
                        # print(f'Trailing for {trail_profit} profit...\n')

                    else:
                        pass
                        # print(f"Failed to close position. {trade_request}Error:", result.comment)


def nap(time_length: int):
    start_time = datetime.now()
    while True:

        trade_management()

        # End time
        end_time = datetime.now()

        # Calculate the duration in seconds
        sleep_duration = (end_time - start_time).total_seconds()
        # print(duration)
        if sleep_duration >= timer:
            break

    return None


def calculate_cooldown_time(base_cooldown=30, exponential_factor=3):
    to_date_ = datetime.now()
    today = to_date_.date()
    # Calculate yesterday's date
    yesterday = today - timedelta(days=1)
    from_date_ = datetime.combine(yesterday, time())

    # Get the last couple of closed positions
    deals = mt5.history_deals_get(from_date_, to_date_)

    deals = deals[::-1]
    consecutive_losses = 0

    # Start from the last closed position and count consecutive losses
    # print(deals)
    for deal in deals:

        # print(deal, deal.profit < 0)
        if deal.profit < 0:
            consecutive_losses += 1

        elif deal.profit > 0:
            # print(deal)
            # Stop counting consecutive losses when a profit is made
            # cooldown_time_seconds = 0
            break
        else:
            continue

    # Calculate cooldown time based on a mathematical formula
    cooldown_time_seconds = base_cooldown * (exponential_factor ** (consecutive_losses - 1))

    if consecutive_losses == 0 or True:
        cooldown_time_seconds = consecutive_losses
    print('Consecutive Losses:', consecutive_losses)
    print(
        f"Cooldown Time: {int(cooldown_time_seconds // 86400)} days, {int((cooldown_time_seconds % 86400) // 3600)} hours, {int((cooldown_time_seconds % 3600) // 60)} minutes, {int(cooldown_time_seconds % 60)} seconds")
    return int(cooldown_time_seconds)


if __name__ == '__main__':
    print('starting...')

    # Initialize MetaTrader 5
    if not mt5.initialize():
        print("initialize() failed, error code =", mt5.last_error())
        quit()

    # (SYMBOL, SL/TP_MULTIPLIER, SPREAD_LIMIT, VOLUME, PROFIT, STRENGTH)
    h1 = ("BTCUSDm", 1, 100, 0.01, 3, 0.1)
    h2 = ("XAUUSDm", 2, 0.31, 0.01, 3, 1)
    h3 = ("US30m", 1, 9, 0.02, 3, 0.1)
    h4 = ("ETHUSDm", 1, 4.3, 0.1, 3, 1)
    h5 = ("EURUSDm", 1, 9.1, 0.01, 3, 0.1)
    h6 = ("GBPUSDm", 1, 0.0002, 0.01, 3, 10000)
    h7 = ("EURJPYm", 1, 0.03, 0.01, 3, 0.1)
    h8 = ("USDCHFm", 1, 0.002, 0.01, 3, 0.1)
    h9 = ("AUDUSDm", 1, 0.0001, 0.01, 3, 0.1)
    h10 = ("USDJPYm", 1, 0.02, 0.01, 3, 0.1)

    all_headers = [h1, h2]
    # all_headers = [h1, h2, h3, h4, h5, h6, h7, h8, h9, h10]
    headers = all_headers.copy()
    sleeping_headers = []

    timer = 0
    incr = 0

    super_sqr = 2 ** 1
    max_positions = len(all_headers)
    long_timer = timer * max_positions
    magic = random.randint(1000, 9999)
    # comment = str(random.randint(100000, 999999))
    random_action = ''
    # random_action_selector = pyip.inputChoice(choices=['buy', 'sell'], blank=True)
    process = None
    scalping_process = None  # For the scalping script (script1)
    position_trading_process = None  # For the opposite script (script2)

    id_checker = False

    last_ticket = {'id': None}

    sl_multiplier = 3  #
    tp_multiplier = 100

    spread_count = 0
    spread_total = 0
    max_spread = 0
    min_spread = 9999

    last_header = None
    last_strength = None
    trendline = ''
    pending_orders = []

    while True:

        positions_orders_total = mt5.positions_total() + mt5.orders_total() + len(pending_orders)

        header = None

        # Open Pending orders
        if len(pending_orders):
            # print(f'{len(pending_orders)} Pending Orders...')

            for instant_request_item in pending_orders:

                # Remove expired orders
                timestamp = instant_request_item['expiration']

                # Convert the timestamp to a datetime object in UTC
                timestamp_dt = datetime.fromtimestamp(timestamp, tz=timezone.utc)

                # Get the current local time and convert it to UTC
                current_time_utc = datetime.now(timezone.utc)

                # Check if the timestamp has expired
                if current_time_utc > timestamp_dt:
                    # Order has expired
                    if instant_request_item in pending_orders:
                        pending_orders.remove(instant_request_item)
                        continue

                #  Run valid orders
                if (instant_request_item['request_type'] == 'buy') and (
                        instant_request_item['price'] <= mt5.symbol_info_tick(symbol).ask):
                    pass
                elif instant_request_item['request_type'] == 'sell' and instant_request_item[
                    'price'] >= mt5.symbol_info_tick(symbol).bid:
                    pass
                else:
                    continue

                instant_result = mt5.order_send(instant_request_item)

                if instant_result and instant_result.retcode != mt5.TRADE_RETCODE_DONE:
                    result_dict = instant_result._asdict()
                    # print(result.retcode)
                    # print(result)
                    if str(instant_result.retcode) == '10044':
                        print(f"Only position closing is allowed for {symbol}")
                        if header in headers:
                            headers.remove(header)
                    else:
                        print(instant_result.comment, instant_request_item['symbol'])
                        # quit()

                    print(f'\nRetrying in {timer}s...\n')
                    nap(timer)
                    continue

                elif not instant_result:
                    print('Error sending instant request!')
                    input('Enter any value to continue | ')
                    continue
                # else:
                #     print(result)

                # print("2. order_send done, ", result)
                print("   opened position with POSITION_TICKET={}".format(instant_result.order))
                # Remove order from pending orders
                if instant_request_item in pending_orders:
                    pending_orders.remove(instant_request_item)

        # if positions_orders_total < len(headers):
        # prioritize unopen symbol
        for header_item in headers:
            # symbol = header[0]
            # mt5.symbol_select(symbol, True)
            symbol_orders = mt5.positions_get(symbol=header_item[0]) + mt5.orders_get(symbol=header_item[0])
            if symbol_orders is None:
                symbol_orders = []
            symbol_orders_array = []
            for order in pending_orders:
                if order['symbol'] == header_item[0]:
                    symbol_orders_array.append(order)

            all_symbol_orders_count = len(symbol_orders_array) + len(symbol_orders)

            # print(all_symbol_orders_count, header_item[0])
            # is_market_volatile = get_atr_data(header_item[0], mt5.TIMEFRAME_H6).iloc[-1]['ATR_Bool']
            is_market_volatile = True
            if is_market_volatile and (all_symbol_orders_count < headers.count(header_item)):
                header = header_item
                break

        margin_level = mt5.account_info().margin_level

        if 0 < margin_level < 500:
            super_trending = True
            nap(timer)
            continue
        else:
            super_trending = False

        if header is None:
            nap(timer)
            continue

        # print(header)
        timer_start_time = datetime.now()

        # if header == last_header:
        #     continue

        symbol = header[0]
        mt5.symbol_select(symbol, True)
        symbol_orders = len(mt5.positions_get(symbol=symbol)) + len(mt5.orders_get(symbol=symbol))

        # ticker = (mt5.symbol_info_tick("ETHUSDm"))
        # print(ticker)
        spread = float(float(mt5.symbol_info_tick(symbol).ask) - float(mt5.symbol_info_tick(symbol).bid))
        point = float(mt5.symbol_info(symbol).point)
        # print(symbol, point)
        # print(spread, point)
        spread_weight = (spread * (1 / point))
        tp = header[1] * spread_weight * tp_multiplier
        sl = header[1] * spread_weight * sl_multiplier
        spread_limit = header[2]

        actions = []
        checklist = []
        while True:
            try:
                # decision = make_trading_decision(symbol, debug=False)
                decision = None
                break
            except:
                print(f'\nError making Trade decision. Retrying in {timer} seconds...')
                nap(timer)

        # strength = round(decision['strength'], 2)
        # action = decision['action']
        # signals = decision['signals']
        # macd_signals = decision['macd_signals']
        # ema_signals = decision['ema_signals']
        # comment = decision['comment']
        # comment = str(comment).replace('.0', '')
        comment = 'BEANS'
        # print(
        # #     f'\nRSI: {signals}\nEMA: {ema_signals}\nMACD: {macd_signals}\nBased on prediction we should be {action}ing {symbol} by {strength}% - #{comment}')
        # if last_strength:
        #     if last_strength > strength:
        #         trendline += '�'
        #     elif last_strength < strength:
        #         trendline += '�'
        #     else:
        #         trendline += '�'
        #     if len(trendline) > 3:
        #         trendline = trendline[1:]
        #
        #     # print(f'Trend: {trendline}')
        #
        # last_strength = strength
        # random_action = pyip.inputChoice(choices=['buy', 'sell'], blank=True)
        # print(f'\nBut instead we will be {random_action}ing!')
        #
        # action = random_action

        point = mt5.symbol_info(symbol).point
        ask_price = mt5.symbol_info_tick(symbol).ask
        bid_price = mt5.symbol_info_tick(symbol).bid

        # print(f'Ask price: {ask_price} | Bid price: {bid_price}')
        spread_count += 1
        spread_total += round(spread, 2)
        if spread > max_spread:
            max_spread = round(spread, 2)
        if spread < min_spread:
            min_spread = round(spread, 2)

        average_spread = round((spread_total / spread_count), 2)
        # print(f'Spread: {spread} | Avg: {average_spread} | Min: {min_spread} | Max: {max_spread}')
        if spread > spread_limit:
            print(f'{symbol} Spread {spread} too large. Auto retry in {timer}s...')
            nap(timer)
            continue
        # print(positions_total)
        # if positions_total >= max_positions:
        if symbol_orders >= headers.count(header):
            # print(
            #     f"You have a {symbol_orders} open positions limit for {symbol}. "
            #     f"We can't open any more. Auto retry in {timer}s...")
            # print('.', end='')
            nap(timer)
            continue

        deviation = 20
        b = account_balance_starter
        a = mt5.account_info().balance
        lot_multiplier = (a // b) + (1 if a % b != 0 else 0)
        lot_multiplier = 1 if lot_multiplier < 1 else lot_multiplier
        lot = header[3] * lot_multiplier

        for action in (['buy', 'sell']):
            if action == 'buy':

                order_type = mt5.ORDER_TYPE_BUY_STOP
                instant_order_type = mt5.ORDER_TYPE_BUY
                price = ask_price
                stop_limit_price = ask_price + (spread * 3)
                # print(f'Buy Price: {price}')
                order_sl = ask_price - sl * point
                order_tp = ask_price + tp * point
            else:

                order_type = mt5.ORDER_TYPE_SELL_STOP
                instant_order_type = mt5.ORDER_TYPE_SELL
                price = ask_price
                stop_limit_price = bid_price - (spread * 3)
                # print(f'Sell Price: {price}')
                order_sl = bid_price + sl * point
                order_tp = bid_price - tp * point

            request = {
                "action": mt5.TRADE_ACTION_PENDING,
                "symbol": symbol,
                "volume": lot,
                "type": order_type,
                "price": stop_limit_price,
                "stoplimit": stop_limit_price,
                "sl": order_sl,
                "tp": order_tp,
                "deviation": deviation,
                "magic": magic,
                "comment": comment,
                "expiration": int((datetime.now(timezone.utc) + timedelta(seconds=150)).timestamp()),
                "type_time": mt5.ORDER_TIME_SPECIFIED,
                "type_filling": mt5.ORDER_FILLING_FOK,
            }

            instant_request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": symbol,
                "volume": lot,
                "type": instant_order_type,
                "request_type": action,
                "price": stop_limit_price,
                "sl": order_sl,
                "tp": order_tp,
                "deviation": deviation,
                "magic": magic,
                "comment": comment,
                "expiration": int((datetime.now(timezone.utc) + timedelta(seconds=150)).timestamp()),
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_FOK,
            }

            timer_end_time = datetime.now()

            # Calculate the duration in seconds
            duration = (timer_end_time - timer_start_time).total_seconds()

            if duration > 5:
                print(f'Duration greater than 5secs - {duration}secs - Retrying...')
                continue
                # if not last_pos['profit']:

            # send a trading request

            # pending_orders.append(instant_request)
            result = mt5.order_send(request)
            if result and result.retcode != mt5.TRADE_RETCODE_DONE:
                result_dict = result._asdict()
                # print(result.retcode)
                # print(result)
                if str(result.retcode) == '10044':
                    print(f"Only position closing is allowed for {symbol}")
                    if header in headers:
                        headers.remove(header)
                elif str(result.retcode) == '10033':
                    # print(f"Pending orders limit of {mt5.account_info().limit_orders} reached")
                    pending_orders.append(instant_request)
                    print(f"Sent order to internal pending orders...")

                else:
                    print(result.comment, symbol)
                    # quit()

                print(f'\nRetrying in {timer}s...\n')
                nap(timer)
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
            nap(timer)

    # Shutdown MetaTrader 5
    mt5.shutdown()
