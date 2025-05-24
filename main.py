from fastapi import FastAPI
import yfinance as yf
import pandas as pd
import lightgbm as lgb
from datetime import datetime
from sklearn.metrics import mean_squared_error
import numpy as np

app = FastAPI()

fang_symbols = ["AAPL", "AMZN", "GOOGL", "META", "MSFT", "NFLX", "NVDA", "TSLA", "SNOW", "AMD"]

def predict_all_stocks():
    messages = []
    total_last_close = 0
    total_pred_close = 0

    for symbol in fang_symbols:
        try:
            # データ取得
            df = yf.download(symbol, period="60d", interval="1d")
            if df.empty:
                messages.append(f"{symbol}: データ取得エラー")
                continue

            df = df.reset_index()
            df["close"] = df["Close"]
            df["close_lag1"] = df["close"].shift(1)
            df = df.dropna()

            X = df[["close_lag1"]]
            y = df["close"]

            split_idx = int(len(df) * 0.8)
            X_train, X_val = X[:split_idx], X[split_idx:]
            y_train, y_val = y[:split_idx], y[split_idx:]

            model = lgb.LGBMRegressor(n_estimators=1000)

            # early stopping コールバック
            early_stopping = lgb.early_stopping(stopping_rounds=50, verbose=False)

            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                callbacks=[early_stopping]
            )

            y_pred = model.predict(X_val)
            val_error = np.sqrt(mean_squared_error(y_val, y_pred))

            last_close = df["close"].iloc[-1]
            X_pred = pd.DataFrame({"close_lag1": [last_close]})
            pred_close = model.predict(X_pred)[0]
            trend = "↑" if pred_close > last_close else "↓"

            messages.append(
                f"{symbol}: 現在 {last_close:.2f} → 予測 {pred_close:.2f} {trend} "
                f"(誤差: {val_error:.2f}, 学習回数: {model.best_iteration_})"
            )

            total_last_close += last_close
            total_pred_close += pred_close

        except Exception as e:
            messages.append(f"{symbol}: エラー発生 ({str(e)})")

    overall_trend = "↑" if total_pred_close > total_last_close else "↓"
    messages.append(
        f"\nFANG+ Index Summary:\n現終値合計: {total_last_close:.2f}\n予測終値合計: {total_pred_close:.2f}\n方向: {overall_trend}"
    )

    return "\n".join(messages)

@app.get("/predict_all")
def get_predictions():
    result = predict_all_stocks()
    return {"message": result}
