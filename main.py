# 必要なライブラリをインポート
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from scipy.optimize import minimize
from simulations.stock_simulation import get_stock_data

# Yahoo Financeから株価データを取得
tickers = [
    "AAPL",   # Apple
    "MSFT",   # Microsoft
    "GOOGL",  # Google
    "AMZN",   # Amazon
    "META",   # Meta (Facebook)
    "NVDA",   # NVIDIA
    "JPM",    # JPMorgan Chase
    "JNJ",    # Johnson & Johnson
    "PG",     # Procter & Gamble
    "KO"      # Coca-Cola
]  # 取得する株式のティッカーを指定
start_date = "2021-01-01"  # 取得開始日
end_date = "2023-12-31"  # 取得終了日
df = get_stock_data(tickers, start_date, end_date)  # 株価データの取得

# リターンとボラティリティ（リスク）の計算
def calculate_returns_and_volatility(df):
    summary = {}
    for column in df.columns:
        log_returns = np.log(df[column] / df[column].shift(1)).dropna()  # 対数リターンを計算
        annual_return = log_returns.mean() * 252  # 年率リターン
        annual_volatility = log_returns.std() * np.sqrt(252)  # 年率ボラティリティ
        summary[column] = {
            "return": annual_return,  # リターン
            "volatility": annual_volatility  # ボラティリティ
        }
    return summary

summary = calculate_returns_and_volatility(df)
print(summary)

# ポートフォリオのリターンを計算
def portfolio_return(weights, df):
    portfolio_price = (df * weights).sum(axis=1)  # ポートフォリオの価格
    log_returns = np.log(portfolio_price / portfolio_price.shift(1)).dropna()  # 対数リターン
    return log_returns.mean() * 252  # 年率リターン

# ポートフォリオのボラティリティ（リスク）を計算
def portfolio_volatility(weights, df):
    portfolio_price = (df * weights).sum(axis=1)  # ポートフォリオの価格
    log_returns = np.log(portfolio_price / portfolio_price.shift(1)).dropna()  # 対数リターン
    return log_returns.std() * np.sqrt(252)  # 年率ボラティリティ

# 効率的フロンティアの計算
num_assets = len(df.columns)  # 資産の数
weights = np.ones(num_assets) / num_assets  # 初期のポートフォリオ（均等配分）
bnds = tuple((0, 1) for _ in range(num_assets))  # 資産の重みは0から1の範囲
min_cagr, max_cagr = min(value["return"] for value in summary.values()), max(value["return"] for value in summary.values())  # 最小・最大のリターン
target_cagr_list = np.linspace(min_cagr, max_cagr, 100)  # リターンの目標範囲を作成

volatilities = []  # 各目標リターンに対応するボラティリティを格納

for target_cagr in target_cagr_list:
    constraints = [
        {"type": "eq", "fun": lambda weights: np.sum(weights) - 1},  # 重みの合計は1
        {"type": "eq", "fun": lambda weights: portfolio_return(weights, df) - target_cagr},  # 目標リターンに一致
    ]

    result = minimize(
        portfolio_volatility,  # ボラティリティを最小化
        weights,  # 初期の重み
        args=(df,),  # 引数（データフレーム）
        method="SLSQP",  # 最適化方法
        bounds=bnds,  # 重みの制約
        constraints=constraints,  # リターン制約
    )

    volatilities.append(result.fun)  # 最適化されたボラティリティをリストに追加

# 結果をプロット
plt.figure(figsize=(15, 8))
plt.plot(volatilities, target_cagr_list, color='red', label='Efficient Frontier')  # 効率的フロンティアを線で表示
plt.scatter([value["volatility"] for value in summary.values()], [value["return"] for value in summary.values()], color='blue', label='Assets')  # 各資産

# ラベルを付ける
for i, asset in enumerate(summary.keys()):
    plt.annotate(asset, (summary[asset]["volatility"], summary[asset]["return"]))  # 資産のラベルを表示

plt.title('Efficient Frontier')  # グラフタイトル
plt.xlabel('Volatility')  # x軸ラベル
plt.ylabel('Return')  # y軸ラベル
plt.legend()  # 凡例
plt.grid()  # グリッド線
plt.show()  # グラフ表示
