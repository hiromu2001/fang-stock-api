from fastapi import FastAPI
import requests
import pandas as pd
import lightgbm as lgb
from datetime import datetime
from sklearn.metrics import mean_squared_error
import numpy as np  # RMSEの手動計算用に追加

app = FastAPI()

API_KEY = "F0NDPFASE93V2QEV"

fang_symbols = ["AAPL", "AMZN", "GOOGL", "META", "MSFT", "NFLX", "NVDA", "TSLA", "SNOW", "AMD"]

def predict_all_stocks():
    messages = []
    total_last_close = 0
    total_pred_close = 0

    for symbol in fang_symbols:
        # API取得
        url = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={symbol}&apikey={API_KEY}"
        response = requests.get(url)
        data = response.json()
        time_series = data.get("Time Series (Daily)")

        if not time_series:
            messages.append(f"{symbol}: データ取得エラー")
            continue

        df = pd.DataFrame.from_dict(time_series, orient="index").astype(float)
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()

        df["close"] = df["4. close"]
        df["close_lag1"] = df["close"].shift(1)
        df = df.dropna()

        X = df[["close_lag1"]]
        y = df["close"]

        split_idx = int(len(df) * 0.8)
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]

        model = lgb.LGBMRegressor(n_estimators=100)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)

        # 修正箇所: RMSEを自分で計算
        val_error = np.sqrt(mean_squared_error(y_val, y_pred))

        last_close = df["close"].iloc[-1]
        X_pred = pd.DataFrame({"close_lag1": [last_close]})
        pred_close = model.predict(X_pred)[0]
        trend = "↑" if pred_close > last_close else "↓"

        messages.append(
            f"{symbol}: 現在 {last_close:.2f} → 予測 {pred_close:.2f} {trend} "
            f"(誤差: {val_error:.2f}, 学習回数: {model.n_estimators_})"
        )

        total_last_close += last_close
        total_pred_close += pred_close

    overall_trend = "↑" if total_pred_close > total_last_close else "↓"
    messages.append(
        f"\nFANG+ Index Summary:\n現終値合計: {total_last_close:.2f}\n予測終値合計: {total_pred_close:.2f}\n方向: {overall_trend}"
    )

    return "\n".join(messages)

@app.get("/predict_all")
def get_predictions():
    result = predict_all_stocks()
    return {"message": result}