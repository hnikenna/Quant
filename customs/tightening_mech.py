def calculate_linear_scaling_percentage(profit, max_profit=3):
    # max_profit = 3
    weight = profit / max_profit  # Normalize the profit to a scale of 0 to 1

    scale = 5
    opp = 100 - scale
    # Map the normalized weight to the percentage scale between 25% and 75%
    scaling_percentage = scale + (opp - scale) * weight

    return min(max(scaling_percentage, scale), opp)


def dynamic_tp_adjustment(current_price, scaled_tp, tp_adjust_threshold=10, tp_increase_amount=20):

    print(scaled_tp * (1 - tp_adjust_threshold / 100))
    # Dynamic TP adjustment: If the current price is within tp_adjust_threshold of the TP, increase TP
    if current_price >= scaled_tp * (1 - tp_adjust_threshold / 100):

        scaled_tp = current_price + (tp_increase_amount/100)
    return scaled_tp


# Example usage:
current_profit = 2.54  # Replace with your logic to calculate current profit
scaled_tp = 3

print(current_profit)
# Calculate the linearly scaled SL percentage
scaling_percentage = calculate_linear_scaling_percentage(current_profit)

# Dynamically calculate SL based on the scaling percentage
scaled_sl = current_profit * (scaling_percentage / 100)

# Dynamically adjust the TP
scaled_tp = dynamic_tp_adjustment(current_profit, scaled_tp)

# Modify the existing position with the dynamically calculated SL and TP
print(scaled_sl, scaled_tp)
