# def calculate_normalized_profit(shares, price, tic):
#     """
#     Calculate normalized profit for a given trade.
#     """
#     if shares == 0:
#         return 0.0
#     # Calculate profit by matching shares from the position queue (FIFO)
#     queue = self.position_queues[tic]
#     shares_to_match = shares
#     total_cost = 0.0
#     total_initial_value = 0.0
#     for idx, (s, p, f) in enumerate(queue):
#         if shares_to_match <= 0:
#             break
#         matched_shares = min(s, shares_to_match)
#         total_cost += matched_shares * price
#         total_initial_value += matched_shares * p
#         shares_to_match -= matched_shares
#     if total_initial_value == 0 or shares_to_match > 0:
#         return 0.0
#     profit = total_cost - total_initial_value
#     return profit / total_initial_value if total_initial_value != 0 else 0.0


# def reward_function(self):
#     """
#     Calculate the reward based on the current portfolio value.
#     """

# current_date = self._get_current_date()
# df_day = self.data[self.data["timestamp"] == current_date]
# prices = df_day.set_index("symbol")["close"].reindex(self.unique_symbols).values

# # Calculate portfolio value
# valid_prices = np.nan_to_num(prices)
# portfolio_value = self.cash + np.sum(valid_prices * self.stock_owned)

# # Calculate normalized profit for each stock
# normalized_profits = [
#     self.calculate_normalized_profit(shares, price, symbol)
#     for shares, price, symbol in zip(
#         self.stock_owned, valid_prices, self.unique_symbols
#     )
# ]

# # Reward is the change in portfolio value from the last step
# if len(self.asset_memory) > 1:
#     previous_value = self.asset_memory[-2]
#     reward = (
#         (portfolio_value - previous_value) / previous_value
#         if previous_value != 0
#         else 0.0
#     )
# else:
#     reward = 0.0

# # Update asset memory
# self.asset_memory.append(portfolio_value)

# return reward + np.mean(normalized_profits)
