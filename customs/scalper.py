import MetaTrader5 as mt5
import pandas as pd
import random, time
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
    global timer
    while True:
        if not mt5.initialize():
            print("initialize() failed, error code =",mt5.last_error())
            mt5.shutdown()
            # quit()
            time.sleep(timer)
        else:
            return True


# get all open positions
# profit = pyip.inputFloat('How much profit are you looking to collect? | ')
profit = -100.05
profit -= 0.01
profit_dict = {}
max_profit_dict = {}
max_profit = 0

if profit < 0.01:
    print(f'Profit too low tf? Enter any value to proceed with a profit of {profit}.')
    input()
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
            # print(position)
            # quit()

            profit_dict[position.ticket] = position.profit
            # print(max_profit_dict)
            try:
                max_profit = max_profit_dict[position.ticket]
            except:
                max_profit = 0

            if position.profit > 0:
                print(position.profit)
                # print(position)
            if max_profit < position.profit > profit:
                print('yes')
                max_profit = position.profit
                max_profit_dict[position.ticket] = position.profit
                print(f'{position.symbol} max profit {max_profit}')

            # print(max_profit)
            if profit < position.profit < max_profit:
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

                    max_profit = 0
                    max_profit_dict[position.ticket] = 0
                    del max_profit_dict[position.ticket]
                    del profit_dict[position.ticket]
                    print(f"Position closed successfully.")
                    print(f'\nScanning for {profit} profit...')

                else:
                    print(f"Failed to close position. {trade_request} - Error:", result.comment)

                    # Remember to shut down the connection to the MT5 terminal
                    # mt5.shutdown()

