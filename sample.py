import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# CoinMarketCapからETH, BTC, SOLの日ごとの最安値データを取得または再利用
base_url ="https://pro-api.coinmarketcap.com/v1/cryptocurrency/ohlcv/historical"
api_key ='dfc295f8-5275-4dbb-a2ac-2186abaceba1'  # あなたのAPIキーをここに入力
symbols = ["ETH", "BTC", "SOL"]
start_date = "2021-01-01"
end_date = "2023-12-31"

data = {}

# データ取得
for symbol in symbols:
    params = {
        'symbol': symbol,
        'time_start': start_date,
        'time_end': end_date,
        'interval': 'daily',
    }
    headers = {
        'Accepts': 'application/json',
        'X-CMC_PRO_API_KEY': api_key,
    }
    
    response = requests.get(base_url, headers=headers, params=params)
    if response.status_code == 200:
        data[symbol] = response.json()['data']['quotes']
    else:
        print(f"Error fetching data for {symbol}: {response.status_code}")

# データをDataFrameに変換
dfs = {}
for symbol, quotes in data.items():
    df = pd.DataFrame(quotes)
    quotes_df = df['quote'].apply(pd.Series)
    quotes_df['date'] = pd.to_datetime(df['time_open']).dt.date
    quotes_df = quotes_df.set_index('date')
    quotes_df['close'] = quotes_df['USD'].apply(lambda x: x['close'])  # 'close' 値を抽出
    dfs[symbol] = quotes_df[['close']]  # 'close' カラムのみを保持

# リターンとボラティリティの計算
returns = []
volatility = []
assets = []

for symbol in dfs:
    daily_returns = dfs[symbol]['close'].pct_change().dropna()
    returns.append(daily_returns.mean())
    volatility.append(daily_returns.std())
    assets.append(symbol)

# ランダムポートフォリオのリターンとボラティリティを計算
random_portfolio_return = np.mean(returns)
random_portfolio_volatility = np.std(returns)

# プロット
plt.figure(figsize=(10, 6))
plt.scatter(volatility, returns, color='blue')

# ラベル付け
for i, asset in enumerate(assets):
    plt.annotate(asset, (volatility[i], returns[i]), fontsize=9)

# ランダムポートフォリオのプロット
plt.scatter(random_portfolio_volatility, random_portfolio_return, color='red', label='random_portfolio')
plt.annotate('random_portfolio', (random_portfolio_volatility, random_portfolio_return), fontsize=9)

# 軸ラベルとタイトル
plt.title('Risk vs Return')
plt.xlabel('Volatility')
plt.ylabel('Return')
plt.legend()
plt.grid()
plt.show()
