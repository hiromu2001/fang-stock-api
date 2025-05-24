from fastapi import FastAPI
import yfinance as yf
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import mean_squared_error
import numpy as np
from sklearn.preprocessing import StandardScaler

app = FastAPI()

@app.get("/predict/{symbol}")
def predict_stock(symbol: str):
    try:
        # データ取得
        df = yf.download(symbol, period="2y")
        if df.empty or len(df) < 60:
            return {"error": f"データが少なすぎるか取得できませんでした: {symbol}"}

        # MultiIndexの解除
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = ['_'.join(col).strip() for col in df.columns.values]

        # 必要な列を抽出
        close_cols = [col for col in df.columns if 'close' in col.lower()]
        volume_cols = [col for col in df.columns if 'volume' in col.lower()]
        if not close_cols:
            return {"error": f"Close列が見つかりません: {symbol}"}
        if not volume_cols:
            return {"error": f"Volume列が見つかりません: {symbol}"}

        close_col = close_cols[0]
        volume_col = volume_cols[0]

        df = df.reset_index()[['Date', close_col, volume_col]]
        df = df.rename(columns={close_col: 'close', volume_col: 'volume'})

        # 対数変換 (出来高・価格×出来高)
        df['log_volume'] = np.log1p(df['volume'])  # log1p(x) = log(1+x)
        df['price_volume'] = df['close'] * df['volume']
        df['log_price_volume'] = np.log1p(df['price_volume'])

        # その他の特徴量
        df['close_lag1'] = df['close'].shift(1)
        df['close_lag2'] = df['close'].shift(2)
        df['ma5'] = df['close'].rolling(5).mean()
        df['ma10'] = df['close'].rolling(10).mean()
        df['return'] = df['close'].pct_change()
        df['volume_lag1'] = df['volume'].shift(1)
        df['volume_ma5'] = df['volume'].rolling(5).mean()
        df['volatility'] = df['return'].rolling(5).std()
        df['volatility_10'] = df['return'].rolling(10).std()

        # 欠損値を削除
        df = df.dropna()

        # 標準化
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(df[['return', 'volatility', 'volatility_10']])
        df[['return_scaled', 'volatility_scaled', 'volatility_10_scaled']] = scaled_features

        # 特徴量リスト
        feature_cols = [
            'close_lag1', 'close_lag2', 'ma5', 'ma10', 'return_scaled',
            'volume_lag1', 'volume_ma5', 'volatility_scaled', 'volatility_10_scaled',
            'log_volume', 'log_price_volume'
        ]

        X = df[feature_cols].copy()
        X.columns = [f"feature_{i}" for i in range(X.shape[1])]
        y = df['close']

        # データ分割
        split_idx = int(len(df) * 0.8)
        X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_val = y.iloc[:split_idx], y.iloc[split_idx:]

        if len(X_train) < 30 or len(X_val) < 30:
            return {"error": f"学習データが少なすぎます: {symbol}"}

        # モデル学習
        model = lgb.LGBMRegressor(
            n_estimators=5000,
            learning_rate=0.01,
            random_state=42
        )

        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            eval_metric='rmse',
            callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
        )

        # 予測
        last_row = df.iloc[-1]
        X_pred = pd.DataFrame({X.columns[i]: [last_row[feature_cols[i]]] for i in range(len(feature_cols))})
        pred_close = model.predict(X_pred)[0]

        val_pred = model.predict(X_val)
        rmse = np.sqrt(mean_squared_error(y_val, val_pred))
        trend = "↑" if pred_close > last_row['close'] else "↓"

        return {
            "symbol": symbol,
            "現在の終値": round(last_row['close'], 2),
            "予測終値": round(pred_close, 2),
            "方向": trend,
            "RMSE（誤差）": round(rmse, 2),
            "学習回数": int(model.best_iteration_) if model.best_iteration_ is not None else model.n_estimators
        }

    except Exception as e:
        return {"error": str(e)}
