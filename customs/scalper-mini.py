import MetaTrader5 as mt5
import pandas as pd
import random
pd.set_option('display.max_columns', 500) # number of columns to be displayed
pd.set_option('display.width', 1500)      # max table width to display
# display data on the MetaTrader 5 package
print("MetaTrader5 package author: ",mt5.__author__)
print("MetaTrader5 package version: ",mt5.__version__)
print()
# establish connection to the MetaTrader 5 terminal
if not mt5.initialize():
    print("initialize() failed, error code =",mt5.last_error())
    quit()


# restart terminal
def restarter():
    # global timer
    mt5.shutdown()
    while True:
        if not mt5.initialize():
            print("initialize() failed, error code =",mt5.last_error())
            # quit()
            # sleep(timer)
        else:
            return True


# get all open positions
profit = 0.05
# profit = 1.05 / 2
profit -= 0.01
max_profit_dict = {}
max_profit = 0

print(f'Scanning for {profit} profit...')
while True:
    positions=mt5.positions_get()
    if positions==None:
        print("No {} positions, error code={}".format(mt5.positions_get(), mt5.last_error()))
        restarter()
    elif len(positions) > 0:
        # print("Total positions on ETHUSDm =",len(positions))
        # display all open positions
        for position in positions:
            # print(position)
            # quit()

            # print(position.profit)

            if position.profit > 0:
                print(position.profit)
                # print(position)
            if max_profit < position.profit > profit:
                max_profit = position.profit
                print(f'max profit {max_profit}')

            if profit < position.profit < max_profit:
                max_profit = 0
                print('-profit hehe!')

                # Prepare a trade request to close the position
                if position.sl == 0:
                    print('No Stop loss for this position')
                    continue
                order_type = mt5.ORDER_TYPE_SELL if position.price_open > position.sl else mt5.ORDER_TYPE_BUY


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
                    print(f"Position closed successfully.")
                    print(f'\nScanning for {profit} profit...')

                else:
                    print(f"Failed to close position. {trade_request}Error:", result.comment)

                    # Remember to shut down the connection to the MT5 terminal
                    mt5.shutdown()

