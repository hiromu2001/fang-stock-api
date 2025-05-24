from fastapi import FastAPI
import yfinance as yf
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import mean_squared_error
import numpy as np

app = FastAPI()

@app.get("/predict/{symbol}")
def predict_stock(symbol: str):
    try:
        # データ取得
        df = yf.download(symbol, period="2y")
        if df.empty:
            return {"error": f"データが取得できませんでした: {symbol}"}

        # 必要な列をシンプル化
        df = df.reset_index()[['Date', 'Close']]
        df = df.rename(columns={'Close': 'close'})
        df['close_lag1'] = df['close'].shift(1)
        df = df.dropna()

        # 特徴量と目的変数
        X = df[['close_lag1']]
        y = df['close']

        # データ分割 (ホールドアウト)
        split_idx = int(len(df) * 0.8)
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]

        # モデル学習（過学習対策あり）
        model = lgb.LGBMRegressor(
            n_estimators=5000,
            learning_rate=0.01,
            random_state=42
        )
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            eval_metric='rmse',
            callbacks=[lgb.early_stopping(stopping_rounds=50)]
        )

        # 予測
        last_close = df['close'].iloc[-1]
        X_pred = pd.DataFrame({'close_lag1': [last_close]})
        pred_close = model.predict(X_pred)[0]

        # 誤差（RMSE）
        val_pred = model.predict(X_val)
        rmse = np.sqrt(mean_squared_error(y_val, val_pred))

        trend = "↑" if pred_close > last_close else "↓"

        return {
            "symbol": symbol,
            "現在の終値": round(last_close, 2),
            "予測終値": round(pred_close, 2),
            "方向": trend,
            "RMSE（誤差）": round(rmse, 2),
            "学習回数": model.best_iteration_
        }

    except Exception as e:
        return {"error": str(e)}
