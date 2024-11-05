import random, math
import MetaTrader5 as mt5
from datetime import datetime, timedelta, time
import subprocess
from time import sleep
from datetime import datetime
from collections import Counter
# import pyautogui as pig

# for i in range(7):
#     print(random.randint(10, 25))

import random


# print(random.randint(10, 25))


# Initialize MetaTrader 5
if not mt5.initialize():
    print("initialize() failed, error code =", mt5.last_error())
    quit()


if True:

    # global last_ticket, id_checker
    # get the number of deals in history

    count_limit = 60

    count = 0
    today = datetime.now().date()
    # Calculate yesterday's date
    yesterday = today - timedelta(days=14)
    from_date = datetime.combine(yesterday, time())
    # from_date = datetime(2024,1,1)

    # get deals for symbols
    if True:
    # while count <= count_limit:
    #     count += 1

        to_date = datetime.now()
        deals = mt5.history_deals_get(from_date, to_date)
        # print(deals[-1])
        the_deal = None

        # print('Last Deal:', deals[-1])
        # print('Last Deal2:', deals[-2])
        # print('Last Deal3:', deals[-3])
        # deals = deals[::-1]
        real_deals = []
        if deals == None:
            print("No deals, error code={}".format(mt5.last_error()))
        elif len(deals) > 0:
            for deal in deals:
                if deal.profit != 0.0:
                    profit = deal.profit
                # print(deal.profit, deal.comment)

                if 'M' in deal.comment:
                    try:
                        if deal.type == 1:
                            deal_type = 'sell'
                        elif deal.type == 0:
                            deal_type = 'buy'
                        else:
                            deal_type = 'err'
                        real_deals.append([deal, profit, deal_type])
                    except:
                        pass

# mt5.shutdown()
# print(real_deals)
real_deals = real_deals[::1]
data = {}
for deal, profit, deal_type in real_deals:
    comment = str(deal.comment).replace('>', '').replace(' ', '')

    if comment.endswith('.'):
        comment += '0'
    # print(comment)
    data[deal.ticket] = [comment, profit, deal_type]

pro_data = []
for key, val in data.items():
    # print(key, val)
    text = val[0]
    result = {}
    group = None

    for pair in text:

        if pair in ['R', 'M', 'L']:
            group = pair
        else:
            try:
                result[group] += str(pair)
            except:
                result[group] = str(pair)
        # print(pair)
    temp_profit = float(val[1])
    # print(temp_profit)
    # # Don't analyze specific losses
    # if temp_profit > 1:
    #     continue
    result['profit'] = temp_profit
    for key, value in result.items():
        result[key] = float(value)
    result['type'] = val[2]
    # print(result)
    pro_data.append(result)

# print(pro_data)

# import pandas as pd
#
# df = pd.DataFrame(pro_data)
# print(df.describe())
# print(df.corr())
#
# import matplotlib.pyplot as plt
# import seaborn as sns
#
# # Create a 3D scatter plot
# fig = plt.figure(figsize=(10, 8))
# ax = fig.add_subplot(111, projection='3d')
#
# # Plotting the dataset
# ax.scatter(df['L'], df['R'], df['profit'], label='Profit')
#
# # Adding labels and legend
# ax.set_xlabel('L-axis')
# ax.set_ylabel('R-axis')
# ax.set_zlabel('Profit-axis')
# ax.set_title('3D Scatter Plot for Profit')
#
# # Show the plot
# plt.show()
# # sns.pairplot(df[['R', 'M', 'E', 'L', 'profit']])
# # plt.show()
#
# sns.scatterplot(x='R', y='profit', data=df)
# plt.show()
# #
# # sns.scatterplot(x='R', y='profit', data=df)
# # plt.show()
# quit()
# # plt.hist(df['profit'], bins=20, edgecolor='black')
# # plt.title('Distribution of Profit')
# # plt.xlabel('Profit')
# # plt.ylabel('Frequency')
# # plt.show()
#
# # sns.barplot(x='E', y='profit', data=df)
# # plt.title('Average Profit by EMA Signal')
# # plt.show()
#
# sns.lineplot(x='R', y='L', data=df)
# plt.title('L Trend by R Signal')
# plt.show()
#
#
# sns.boxplot(x='R', y='L', data=df)
# plt.title('Box Plot of Profit by RSI Signal and Linear Regression')
# plt.show()
# #
# # correlation_matrix = df[['R', 'M', 'E', 'L', 'profit']].corr()
# # sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
# # plt.title('Correlation Heatmap')
# # plt.show()
#
#
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LinearRegression
#
# X = df[['R', 'M', 'E', 'L']]
# y = df['profit']
#
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#
# model = LinearRegression()
# model.fit(X_train, y_train)
#
# print(f'R-squared score: {model.score(X_test, y_test)}')
#
#
# last_header = None
# last_strength = None
# trendline = ''
#

# print(data)
multipliers = [0.5, 1, 2, -0.5, -1, -2]
max_profit = 0
max_profit_data = {}
formula_data = {}
for i in multipliers:
    for j in multipliers:
        for k in multipliers:

            formula = f'(R * {i}) + (M * {j}) + (L * {k})'
            profits = 0
            losses = 0

            for data in pro_data:
                result = (data['R'] * i) + (data['M'] * j) + (data['L'] * k)

                if result > 0:
                    action = 'buy'
                elif result < 0:
                    action = 'sell'
                else:
                    # print(f'Error: {data} // {formula} // {result}')

                    if data['profit'] > 0:
                        losses += 1
                    elif data['profit'] < 0:
                        profits += 1
                    continue

                if action == data['type']:
                    if data['profit'] > 0:
                        profits += 1
                    elif data['profit'] < 0:
                        losses += 1
                else:
                    if data['profit'] > 0:
                        losses += 1
                    elif data['profit'] < 0:
                        profits += 1

            formula_profit = ((profits - losses) / (profits + losses))
            # print(f'Profits: {profits}, Losses: {losses}, Percentage: {formula_profit * 100}%')
            formula_data[formula] = formula_profit

            if formula_profit > max_profit:
                max_profit = formula_profit
                max_profit_data = {'Formula': formula, 'Win-Rate(%)': round((formula_profit * 100), 2)}
                print(max_profit_data)

# print(formula_data)
