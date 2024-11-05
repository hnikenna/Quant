import MetaTrader5 as mt5
import pandas as pd
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
 
# display data on active orders on EURUSDm
orders=mt5.orders_get(symbol="BTCUSDm")
if orders is None:
    print("No orders on EURUSDm, error code={}".format(mt5.last_error()))
else:
    print("Total orders on EURUSDm:",len(orders))
    # display all active orders
    for order in orders:
        print(order)
quit()
 
# get the list of orders on symbols whose names contain "*BTC*"
BTC_orders=mt5.orders_get(group="*XAU*")
if BTC_orders is None:
    print("No orders with group=\"*BTC*\", error code={}".format(mt5.last_error()))
else:
    print("orders_get(group=\"*BTC*\")={}".format(len(BTC_orders)))
    # display these orders as a table using pandas.DataFrame
    df=pd.DataFrame(list(BTC_orders),columns=BTC_orders[0]._asdict().keys())
    df.drop(['time_done', 'time_done_msc', 'position_id', 'position_by_id', 'reason', 'volume_initial', 'price_stoplimit'], axis=1, inplace=True)
    df['time_setup'] = pd.to_datetime(df['time_setup'], unit='s')
    print(df)
 
# shut down connection to the MetaTrader 5 terminal
mt5.shutdown()
