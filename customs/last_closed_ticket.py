import MetaTrader5 as mt5
from datetime import datetime, timedelta, time
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

# get the number of deals in history

today = datetime.now().date()
# Calculate yesterday's date
yesterday = today - timedelta(days=1)
from_date = datetime.combine(yesterday, time())

to_date=datetime.now()

# get deals for symbols whose names contain neither "EUR" nor "GBP"
deals = mt5.history_deals_get(from_date, to_date)
deals = deals[::-1]
if deals == None:
    print("No deals, error code={}".format(mt5.last_error()))
elif len(deals) > 0:
    for deal in deals:
        if deal.reason == 3:
            # print(f'Error with position {deal.position_id} reason')
            continue
        elif deal.reason == 4:
            # Loss
            print(f'You made a loss of {deal.profit}')
            break
        elif deal.reason == 5:
            # Profit
            print(f'You made a profit of {deal.profit}')
            break
        else:
            print(f'Error with position {deal.position_id} reason')
        print("  ",deal)


# shut down connection to the MetaTrader 5 terminal
mt5.shutdown()
