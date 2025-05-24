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
        df = yf.download(symbol, period="2y")
        if df.empty or len(df) < 30:
            return {"error": f"データが少なすぎるか取得できませんでした: {symbol}"}

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = ['_'.join(col).strip() for col in df.columns.values]

        df = df.reset_index()[['Date', 'Close']]
        df = df.rename(columns={'Close': 'close'})
        df['close_lag1'] = df['close'].shift(1)
        df = df.dropna()

        X = df[['close_lag1']]
        # カラム名を完全に安全な名前にリネーム
        X.columns = [f"feature_{i}" for i in range(X.shape[1])]
        y = df['close']

        split_idx = int(len(df) * 0.8)
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]

        if len(X_train) < 10 or len(X_val) < 10:
            return {"error": f"学習データが少なすぎます: {symbol}"}

        model = lgb.LGBMRegressor(
            n_estimators=5000,
            learning_rate=0.01,
            random_state=42
        )
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            eval_metric='rmse',
            callbacks=[lgb.early]()
