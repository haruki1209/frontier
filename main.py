# main.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from scipy.optimize import minimize
from simulations.stock_simulation import get_stock_data

# Yahoo Financeから株価データを取得
tickers = ["AAPL", "MSFT", "GOOGL"]
start_date = "2021-01-01"
end_date = "2023-12-31"
df = get_stock_data(tickers, start_date, end_date)

def calculate_returns_and_volatility(df):
    summary = {}
    for column in df.columns:
        log_returns = np.log(df[column]) - np.log(df[column].shift(1))
        log_returns.dropna(inplace=True)
        annual_return = log_returns.mean() * 252
        annual_volatility = log_returns.std() * np.sqrt(252)
        summary[column] = {
            "return": annual_return,
            "volatility": annual_volatility
        }
    return summary

summary = calculate_returns_and_volatility(df)
print(summary)

def portfolio_return(weights, df):
    portfolio_price = (df * weights).sum(axis=1)
    log_returns = np.log(portfolio_price) - np.log(portfolio_price.shift(1))
    log_returns.dropna(inplace=True)
    annual_return = log_returns.mean() * 252
    return annual_return

def portfolio_volatility(weights, df):
    portfolio_price = (df * weights).sum(axis=1)
    log_returns = np.log(portfolio_price) - np.log(portfolio_price.shift(1))
    log_returns.dropna(inplace=True)
    annual_volatility = log_returns.std() * np.sqrt(252)
    return annual_volatility

# 効率的フロンティアの計算
weights = (1 / len(df.columns)) * np.ones(len(df.columns))
bnds = tuple((0, 1) for x in range(len(df.columns)))
min_cagr, max_cagr = min(value["return"] for value in summary.values()), max(value["return"] for value in summary.values())
target_cagr_list = np.linspace(min_cagr, max_cagr, 100)

volatilities = []
for target_cagr in target_cagr_list:
    constraints = [
        {"type": "eq", "fun": lambda weights: np.sum(weights) - 1},
        {
            "type": "eq",
            "fun": lambda weights: portfolio_return(weights, df) - target_cagr,
        },
    ]

    result = minimize(
        portfolio_volatility,
        weights,
        args=(df,),
        method="SLSQP",
        bounds=bnds,
        constraints=constraints,
    )

    volatilities.append(result["fun"])
volatilities = np.array(volatilities)

# プロット
plt.figure(figsize=(15, 8))
plt.scatter(volatilities, target_cagr_list, color='red', label='Efficient Frontier')
plt.scatter([value["volatility"] for value in summary.values()], [value["return"] for value in summary.values()], label='Assets')

# ラベル付け
for i, asset in enumerate(summary.keys()):
    plt.annotate(asset, (summary[asset]["volatility"], summary[asset]["return"]))

plt.title('Efficient Frontier')
plt.xlabel('Volatility')
plt.ylabel('Return')
plt.legend()
plt.grid()
plt.show()