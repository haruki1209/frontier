# simulations/stock_simulation.py
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

# 変数の定義
mu = 0.0005  # 期待収益率（小さめに設定）
sigma = 0.02  # ボラティリティ（小さめに設定）
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

# より現実的な資産パラメータの設定
assets = [
    {
        "name": "asset" + str(i),
        "param": {
            "mu": np.random.uniform(0.0001, 0.001),  # ランダムな期待収益率
            "sigma": np.random.uniform(0.01, 0.03),  # ランダムなボラティリティ
            "S0": 100,
            "dt": dt,
            "T": 365,
        },
    }
    for i in range(100)
]

# リターンとボラティリティの計算
returns = []
volatilities = []

for asset in assets:
    mu = asset["param"]["mu"]
    sigma = asset["param"]["sigma"]
    # シミュレーションの実行（例としてランダムウォーク）
    prices = [asset["param"]["S0"]]
    for _ in range(asset["param"]["T"]):
        price = prices[-1] * np.exp(np.random.normal(mu * dt, sigma * np.sqrt(dt)))
        prices.append(price)
    
    # 最終価格からリターンとボラティリティを計算
    final_return = (prices[-1] - prices[0]) / prices[0]
    volatility = np.std(prices) / prices[0]
    
    returns.append(final_return)
    volatilities.append(volatility)

# プロット
plt.figure(figsize=(10, 6))
plt.scatter(volatilities, returns, marker='o')  # 線を引かずに散布図を作成
for i, asset in enumerate(assets):
    plt.annotate(asset["name"], (volatilities[i], returns[i]))

plt.title('Return vs Volatility')
plt.xlabel('Volatility')
plt.ylabel('Return')
plt.grid()
plt.show()