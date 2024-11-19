# simulations/stock_simulation.py
import numpy as np
import pandas as pd
import yfinance as yf

# 変数の定義
mu = 0.01  # 期待収益率
sigma = 0.03  # ボラティリティ
dt = 1 / 252  # 時間刻み（1日）

def get_stock_data(tickers, start_date, end_date):
    """
    Yahoo Financeから株価データを取得
    """
    data = yf.download(tickers, start=start_date, end=end_date)
    return data['Adj Close']  # 調整後終値を返す

# 取得する株のティッカーシンボル
tickers = ["AAPL", "MSFT", "GOOGL"]  # 例としてApple, Microsoft, Googleを指定
start_date = "2021-01-01"
end_date = "2023-12-31"

# 株価データを取得
df = get_stock_data(tickers, start_date, end_date)

# データの確認
print(df.head())

# 資産の設定
assets = [
    {
        "name": "asset" + str(i),
        "param": {
            "mu": mu * i,
            "sigma": sigma * i,
            "S0": 100,
            "dt": dt,
            "T": 30,
        },
    }
    for i in range(10)
]