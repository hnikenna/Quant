 import MetaTrader5 as mt5
from time import sleep

# Initialize connection to MetaTrader 5
if not mt5.initialize():
    print("Initialize failed")
    mt5.shutdown()
    exit()

# Get account information
account_info = mt5.account_info()

print(account_info)

if account_info is not None:
    balance = account_info.balance
    equity = account_info.equity

    print(f"Account Equity: {equity}")
    print(f"Account Balance: {balance}")
    print(f"Scalp Threshold: {balance * 2.1}")
    print(f"Leverage: {account_info.leverage}")
else:
    print("Failed to retrieve account information")

while True:
    print(mt5.account_info().margin_level)
    sleep(2)
# Shutdown connection
# mt5.shutdown()

# AccountInfo(login=179777976, trade_mode=0, leverage=2000000000, limit_orders=1024, margin_so_mode=0, trade_allowed=True, trade_expert=True, margin_mode=2, currency_digits=2, fifo_close=False, balance=13.61, credit=0.0, profit=4.84, equity=18.45, margin=2.1, margin_free=16.35, margin_level=878.5714285714284, margin_so_call=60.0, margin_so_so=0.0, margin_initial=0.0, margin_maintenance=0.0, assets=0.0, liabilities=0.0, commission_blocked=0.0, name='Demo', server='Exness-MT5Trial9', currency='USD', company='Exness Technologies Ltd')