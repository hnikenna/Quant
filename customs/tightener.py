import MetaTrader5 as mt5
import pandas as pd
import random, time


pd.set_option('display.max_columns', 500) # number of columns to be displayed
pd.set_option('display.width', 1500)      # max table width to display
# display data on the MetaTrader 5 package
# establish connection to the MetaTrader 5 terminal
if not mt5.initialize():
    print("initialize() failed, error code =",mt5.last_error())
    quit()


# restart terminal
def restarter():
    global timer
    while True:
        if not mt5.initialize():
            print("initialize() failed, error code =",mt5.last_error())
            mt5.shutdown()
            # quit()
            time.sleep(timer)
        else:
            return True


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


def dynamic_tp_adjustment(current_price, scaled_tp, tp_adjust_threshold=50, tp_increase_amount=250):

    # print('Take Profit Threshold for Current Profit:', scaled_tp * (1 - tp_adjust_threshold / 100))
    # Dynamic TP adjustment: If the current price is within tp_adjust_threshold of the TP, increase TP
    if current_price >= scaled_tp * (1 - tp_adjust_threshold / 100):

        scaled_tp = current_price + (tp_increase_amount/100)
    return scaled_tp


# get all open positions
profit = 0.15
profit -= 0.01
timer = 1
# max_profit_dict = {}
# max_profit = 0

print(f'Scanning for {profit} profit...')
while True:
    # sleep(3)
    positions=mt5.positions_get()
    if positions==None:
        print("No {} positions, error code={}".format(mt5.positions_get(), mt5.last_error()))
        restarter()
    elif len(positions) > 0:
        # print("Total positions on ETHUSDm =",len(positions))
        # display all open positions
        for position in positions:

            if profit < position.profit:
                max_profit = 0
                # print('-profit hehe!')

                # Prepare a trade request to close the position
                # if position.sl == 0:
                #     print('No Stop loss for this position')
                #     continue

                order_type = 'buy' if position.type == 0 else 'sell'

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

                # Dynamically adjust the TP
                scaled_tp = dynamic_tp_adjustment(position.profit, get_scaled_tp)
                # print(get_scaled_tp)
                # print('Scaled Tp:', scaled_tp)

                new_tp = position.tp

                # Update tp if scaled one is larger
                if scaled_tp > get_scaled_tp:
                    print('upgrade tp')
                    tp_increment = scaled_tp * pip_scale

                    new_tp = (position.price_open + tp_increment) if order_type == 'buy' else (position.price_open - tp_increment)

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

                # Update sl if scaled one is larger
                if scaled_sl > get_scaled_sl:
                    print('upgrade sl')
                    sl_increment = scaled_sl * pip_scale

                    new_sl = (position.price_open + sl_increment) if order_type == 'buy' else (position.price_open - sl_increment)
                # quit()
                # print('tst')
                if new_tp == position.tp and new_sl == position.sl:
                    # print('No changes..')
                    # time.sleep(timer)
                    continue
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
                    print(f"Position modified successfully.")
                    print(f'\nScanning for {profit} profit...')

                else:
                    print(f"Failed to close position. {trade_request}Error:", result.comment)


                    # Remember to shut down the connection to the MT5 terminal
                    # mt5.shutdown()

                time.sleep(timer)
