import MetaTrader5 as mt5
import pandas as pd
import random, time
from traderPro import make_trading_decision


pd.set_option('display.max_columns', 500) # number of columns to be displayed
pd.set_option('display.width', 1500)      # max table width to display
# display data on the MetaTrader 5 package
# establish connection to the MetaTrader 5 terminal
if not mt5.initialize():
    # print("initialize() failed, error code =",mt5.last_error())
    quit()


# restart terminal
def restarter():
    global timer
    mt5.shutdown()
    while True:
        if not mt5.initialize():
            # print("initialize() failed, error code =",mt5.last_error())
            # quit()
            time.sleep(timer)
        else:
            return True


# get all open positions
timer = 0.5

print(f'Scanning for signals...')
while True:
    # sleep(3)
    positions=mt5.positions_get()
    if positions==None:
        # print("No {} positions, error code={}".format(mt5.positions_get(), mt5.last_error()))
        restarter()
    elif len(positions) > 0:
        # print("Total positions on ETHUSDm =",len(positions))
        # display all open positions
        for position in positions:
            order_type = 'buy' if position.type == 0 else 'sell'
            new_sl = position.sl
            decision = make_trading_decision(symbol=position.symbol, debug=False)
            action = decision['action']
            # action = 'sell'

            # print(order_type, action)
            if position.profit == 0:
                continue
            pip_scale = float(abs((position.price_open - position.price_current) / position.profit))
            # print('Pip Scale:', pip_scale)

            if order_type == 'sell':
                current_profit = float((position.price_open - position.price_current) / pip_scale)
                # print('Current Profit:', current_profit)
                get_scaled_sl = (position.price_open - position.sl) / pip_scale
            else:
                current_profit = float((position.price_current - position.price_open) / pip_scale)
                # print('Current Profit:', current_profit)
                get_scaled_sl = (position.sl - position.price_open) / pip_scale
            # print(position.price_open, position.sl, pip_scale)
            # print(position.price_open - position.sl)
            # print((position.price_open - position.sl) / pip_scale)
            # print(get_scaled_sl)
            # print('SL', get_scaled_sl)
            # print(current_profit - get_scaled_sl)
            # quit()
            adjustment_threshold = 7.0  # Example: Adjust stop loss by 2%

            if order_type != action:

                # Specify a percentage or fixed distance for the adjustment
                adjustment_percentage = max((6 - round(abs(current_profit), 1)), 1.10)
                # adjustment_percentage = 3.0  # Example: Adjust stop loss by 50%

                # Calculate the new stop loss based solely on the current profit
                # scaled_sl = (current_profit + get_scaled_sl) / 2
                if current_profit < 0:
                    # If position is in a loss it should set new stop loss at twice the loss price
                    scaled_sl = current_profit * adjustment_percentage
                else:
                    # If position is in a profit it should set new stop loss at half the profit price
                    scaled_sl = current_profit / adjustment_percentage
                # print('Scaled Sl:', scaled_sl)

                # Update sl if scaled one is larger
                if scaled_sl > get_scaled_sl:
                    print('Opposite signal detected. Tightening Stop Loss...')
                    sl_increment = scaled_sl * pip_scale

                    new_sl = (position.price_open + sl_increment) if order_type == 'buy' else (position.price_open - sl_increment)
                    new_sl = round(new_sl, 2)
            elif (current_profit - get_scaled_sl) <= adjustment_threshold:
                # Specify a percentage or fixed distance for the adjustment
                adjustment_percentage = 2  # Example: Adjust stop loss by 2%
                min_sl_scale = -0.75 * adjustment_percentage # instead of multiplying low profit levels like 0.01 by adj_perc above, we use this minimum instead
                # Calculate the new stop loss based solely on the current profit
                if current_profit < 0:
                    # If position is in a loss it should set new stop loss at twice the loss price
                    scaled_sl = min(current_profit * adjustment_percentage, adjustment_percentage)
                    # print('Scaled Sl:', scaled_sl)


                    # Update sl if scaled one is larger
                    if scaled_sl < get_scaled_sl:
                        print('Price too Close to Stop loss. Widening Stop Loss...')
                        sl_increment = scaled_sl * pip_scale

                        new_sl = (position.price_open + sl_increment) if order_type == 'buy' else (position.price_open - sl_increment)
                        new_sl = round(new_sl, 2)

            if new_sl != position.sl:

                trade_request = {
                    "action": mt5.TRADE_ACTION_SLTP,
                    "symbol": position.symbol,
                    # "volume": position.volume,  # Specify the volume to close
                    # "type": order_type,  # Specify the order type (SELL to close a long position)
                    "position": position.ticket,  # Ticket number of the position to close
                    'sl': new_sl,
                    'tp': position.tp,

                }

                # Send the trade request to close the position
                result = mt5.order_send(trade_request)

                if result.retcode == mt5.TRADE_RETCODE_DONE:
                    print(f"Position modified successfully.")
                    print(f'\nScanning for signals...')

                else:
                    print(f"Failed to close position. {trade_request}Error:", result.comment)


                    # Remember to shut down the connection to the MT5 terminal
                    # mt5.shutdown()

                time.sleep(timer)
