import random

import math


def get_value_for_rtr_increment(rtr_increment):
    # Calculate the number of zeros after the decimal point
    leading_zeros = -math.floor(math.log10(rtr_increment))

    # The return value is the number of leading zeros + 1
    return leading_zeros + 2


count = 0
reset_count = 0
total_profitability = 0
spread_rate = 0.05      # Calculated by the %change of a new trade
min_rtr = 0.1
max_rtr = 1.5
rtr_increment = 0.1
value_for_rtr_increment = get_value_for_rtr_increment(rtr_increment)
risk_reward_rate = min_rtr - rtr_increment
# max_rtr = max_rtr - rtr_increment
max_rate = {'RTR': 0, 'Profitability': 0}
min_rate = {'RTR': 0, 'Profitability': 999999}
rate_data = {}
while True:

    count += 1

    # Simulation parameters
    base_amount = 80
    risk_reward_rate += rtr_increment # Equal TP and SL
    spread = base_amount * spread_rate
    profit_rate = spread * risk_reward_rate
    num_simulations = 10000  # Number of individual simulations to run
    num_iterations = 1000  # Number of steps in each simulation

    # Tracking variables for profitability across simulations
    total_profit = 0
    total_loss = 0

    # Run the simulations
    for _ in range(num_simulations):
        current_value = base_amount
        profit = 0
        loss = 0

        for _ in range(num_iterations):
            current_value += random.choice([-1, 1])

            if current_value <= spread:  # Loss condition
                loss += 1
                current_value = base_amount

            elif current_value >= (base_amount + spread + profit_rate):  # Profit condition
                profit += risk_reward_rate
                current_value = base_amount

            # print(current_value)

        total_profit += profit
        total_loss += loss

    # Calculate average profitability
    average_profitability = (total_profit / total_loss) if total_loss != 0 else float('inf')

    # Output results
    print(f"Total simulations: {num_simulations}")
    print(f"Average total profit per simulation: {round(total_profit / num_simulations, value_for_rtr_increment)}")
    print(f"Average total loss per simulation: {round(total_loss / num_simulations, value_for_rtr_increment)}")
    print(f"Average profitability (profit/loss ratio): {round(average_profitability, value_for_rtr_increment)}")
    # total_profitability += average_profitability
    total_profitability = average_profitability
    count = 1
    total_profitability_percentage = (total_profitability/count)*50
    print(f"Profitability ({round(risk_reward_rate, value_for_rtr_increment)}): {round(total_profitability_percentage, value_for_rtr_increment)}%")

    if total_profitability_percentage > max_rate['Profitability']:
        max_rate['RTR'] = risk_reward_rate
        max_rate['Profitability'] = total_profitability_percentage

    print(f'Best Win: {round(max_rate['RTR'], value_for_rtr_increment)} - {round(max_rate['Profitability'], value_for_rtr_increment)}%')

    if total_profitability_percentage < min_rate['Profitability']:
        min_rate['RTR'] = risk_reward_rate
        min_rate['Profitability'] = total_profitability_percentage

    print(f'Worst Loss: {round(min_rate['RTR'], value_for_rtr_increment)} - {round(min_rate['Profitability'], value_for_rtr_increment)}%')

    try:
        rate_data[round(risk_reward_rate, value_for_rtr_increment)] = (rate_data[round(risk_reward_rate, 1)] + total_profitability_percentage) / 2
    except:
        rate_data[round(risk_reward_rate, value_for_rtr_increment)] = total_profitability_percentage

    if risk_reward_rate >= max_rtr:
        reset_count += 1
        risk_reward_rate = min_rtr - rtr_increment
        for key, value in rate_data.items():
            print(f'[{key} - {round(value, value_for_rtr_increment)}%] ', end='')

        print(f' ({reset_count})\n')
    print('*'*75)


