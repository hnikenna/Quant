import subprocess


while True:

    scalp_process = None
    tight_process = None
    # new_actions = get_actions(timeframes)
    # # print(new_actions)
    #
    # # Scalp if irregularity is detected
    # checklist = [1 if act == 'buy' else 0 for act in new_actions]
    # condition = bool(1 in checklist and 0 in checklist)
    # condition = False
    # condition = mt5.positions_total()
    condition = False
    if not condition:
        # Tighten By Default
        if scalp_process is not None and scalp_process.poll() is None:
            print("No longer Scalping")
            scalp_process.terminate()

        if tight_process is None or tight_process.poll() is not None:
            print("No Volatility! Tightening the positions.")
            tight_process = subprocess.Popen(["python", "tightener.py"])

    else:

        if tight_process is not None or tight_process.poll() is None:
            print("No longer Tightening")
            tight_process.terminate()

        if scalp_process is None and scalp_process.poll() is not None:
            print("Volatility! Scalping the positions.")
            scalp_process = subprocess.Popen(["python", "scalper.py"])