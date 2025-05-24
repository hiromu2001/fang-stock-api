import yfinance as yf
import pandas as pd
import lightgbm as lgb
import numpy as np
from fastapi import FastAPI

app = FastAPI()

fang_symbols = ["AAPL", "AMZN", "GOOGL", "META", "MSFT", "NFLX", "NVDA", "TSLA", "SNOW", "AMD"]

def predict_all_stocks():
    messages = []
    total_last_close = 0
    total_pred_close = 0

    for symbol in fang_symbols:
        try:
            df = yf.download(symbol, period="1y", interval="1d")
            if df.empty:
                messages.append(f"{symbol}: データ取得エラー")
                continue

            # MultiIndex解除
            df.columns = [col if isinstance(col, str) else col[0] for col in df.columns]

            df["close_lag1"] = df["Close"].shift(1)
            df = df.dropna()

            X = df[["close_lag1"]]
            y = df["Close"]

            split_idx = int(len(df) * 0.8)
            X_train, X_val = X[:split_idx], X[split_idx:]
            y_train, y_val = y[:split_idx], y[split_idx:]

            model = lgb.LGBMRegressor(n_estimators=1000)
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=50, verbose=False)

            y_pred = model.predict(X_val)
            val_error = np.sqrt(((y_val - y_pred) ** 2).mean())

            last_close = df["Close"].iloc[-1]
            X_pred = pd.DataFrame({"close_lag1": [last_close]})
            pred_close = model.predict(X_pred)[0]
            trend = "↑" if pred_close > last_close else "↓"

            messages.append(
                f"{symbol}: 現在 {last_close:.2f} → 予測 {pred_close:.2f} {trend} (誤差: {val_error:.2f}, 学習回数: {model.best_iteration_})"
            )

            total_last_close += last_close
            total_pred_close += pred_close

        except Exception as e:
            messages.append(f"{symbol}: エラー発生 ({e})")
            continue

    overall_trend = "↑" if total_pred_close > total_last_close else "↓"
    messages.append(
        f"\nFANG+ Index Summary:\n現終値合計: {total_last_close:.2f}\n予測終値合計: {total_pred_close:.2f}\n方向: {overall_trend}"
    )

    return "\n".join(messages)

@app.get("/predict_all")
def get_predictions():
    result = predict_all_stocks()
    return {"message": result}
