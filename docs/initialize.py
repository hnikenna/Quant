import MetaTrader5 as mt5
import config
# display data on the MetaTrader 5 package
print("MetaTrader5 package author: ",mt5.__author__)
print("MetaTrader5 package version: ",mt5.__version__)

path = "C:\\Program Files\\MetaTrader 5\\terminal64.exe"
# establish MetaTrader 5 connection to a specified trading account
if not mt5.initialize():
# if not mt5.initialize(login=config.login, server=config.server, password=config.password, portable=True, path=path):
    print("initialize() failed, error code =",mt5.last_error())
    quit()

# display data on connection status, server name and trading account
print(mt5.terminal_info())
# display data on MetaTrader 5 version
print(mt5.version())

# shut down connection to the MetaTrader 5 terminal
mt5.shutdown()
