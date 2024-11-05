import MetaTrader5 as mt5
import subprocess, time


if not mt5.initialize():
    print("initialize() failed, error code =",mt5.last_error())
    quit()
process = None
while True:
    # get the list of positions on symbols whose names contain "*USD*"
    usd_positions=mt5.positions_get(group="*USD*")
    checklist = []
    for position in usd_positions:
        # print(' | '.join([str(position.symbol), str(position.profit), str(position.comment), str('sell' if position.type == 1 else 'buy')]))

        if False or 'ETHUSD' in position.symbol:
            checklist.append(position.type)

    # print(checklist)
    condition = bool(1 in checklist and 0 in checklist)

    if condition:
        # If the condition is true and the script is not running, start the script
        if process is None or process.poll() is not None:
            print("Buy and Sell Noticed! Starting the script.")
            process = subprocess.Popen(["python", "scalper-mini.py"])
    else:
        # If the condition becomes false and the script is running, terminate it
        # subprocess.Popen(["python", "scalper-mini.py"]).terminate()
        if process is not None and process.poll() is None:
            print("No buy and Sell notice! Terminating the script.")
            process.terminate()

    time.sleep(0.05)


# shut down connection to the MetaTrader 5 terminal
mt5.shutdown()
