# main.py
from fastapi import FastAPI
import yfinance as yf
import lightgbm as lgb
import pandas as pd
import numpy as np

# FastAPIアプリのインスタンス作成
app = FastAPI()

# 4. FastAPIエンドポイント/predict_allを作成し、各銘柄の予測結果とFANG+合計値の変動方向をJSONで返す
@app.get("/predict_all")
def predict_all():
    # FANG+銘柄リスト
    tickers = ["AAPL", "AMZN", "GOOGL", "META", "MSFT", "NFLX", "NVDA", "TSLA", "SNOW", "AMD"]
    results = []  # 各銘柄の結果を格納するリスト
    
    for ticker in tickers:
        # yfinanceで過去データを取得（過去5年分の終値など）
        df = yf.download(ticker, period="5y")
        # 2. MultiIndexとなった出力からClose列を取得し、列名を'close'にリネーム
        if isinstance(df.columns, pd.MultiIndex):
            close_data = df["Close"]  # Close列を選択
            # データ型に応じてSeriesをDataFrameに変換し列名を設定
            if isinstance(close_data, pd.DataFrame):
                close_df = close_data.copy()
                close_df.columns = ["close"]
            else:
                close_df = close_data.to_frame(name="close")
        else:
            # MultiIndexでない場合も対応（列名を'close'に）
            close_df = df[["Close"]].rename(columns={"Close": "close"})
        
        # 時系列データを用いて翌日の終値(close)を予測するための特徴量と目的変数を作成
        # 現在のclose値から翌日close値（target）を予測する（ホールドアウト検証のため過去データで学習）
        close_df["target"] = close_df["close"].shift(-1)      # 翌日の終値（目的変数）
        close_df["lag1"] = close_df["close"].shift(1)         # 1日前の終値
        close_df["lag2"] = close_df["close"].shift(2)         # 2日前の終値
        close_df["lag3"] = close_df["close"].shift(3)         # 3日前の終値
        
        # 予測に使用する直近データ（最新日の特徴量）を保存しておく
        last_features = close_df[["close", "lag1", "lag2", "lag3"]].iloc[-1]
        current_price = float(last_features["close"])  # 現在値（最新日の終値）
        
        # データから欠損を除去（初期数日間のlag欠損と最後のtarget欠損を落とす）
        data = close_df.dropna()
        X_all = data[["close", "lag1", "lag2", "lag3"]]  # 説明変数
        y_all = data["target"]                          # 目的変数（翌日終値）
        
        # 3. データを8:2でホールドアウト分割（過去データの80%を学習に、20%をテストに利用）
        split_idx = int(len(X_all) * 0.8)
        X_train, X_test = X_all.iloc[:split_idx], X_all.iloc[split_idx:]
        y_train, y_test = y_all.iloc[:split_idx], y_all.iloc[split_idx:]
        
        # 1. LightGBMモデルの設定（n_estimatorsを高めに設定しearly_stopping_roundsで早期終了）
        model = lgb.LGBMRegressor(n_estimators=1000, random_state=0)
        model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            eval_metric="rmse",            # 検証指標としてRMSEを指定
            early_stopping_rounds=50,      # 検証誤差が改善しなくなったら50ラウンドで学習打ち切り
            verbose=False                  # 学習の詳細を出力しない
        )
        # 学習に使用した実際のラウンド数（early stoppingで決定した最適なイテレーション数）
        best_iter = model.best_iteration_ if model.best_iteration_ is not None else model.n_estimators
        
        # テストデータでのRMSEを算出（予測値と実測値の誤差）
        y_pred = model.predict(X_test, num_iteration=best_iter)
        rmse = float(np.sqrt(np.mean((y_pred - y_test.to_numpy())**2)))
        
        # 翌営業日の終値を予測（最新日の特徴量をモデルに入力）
        pred_price = float(model.predict(last_features.to_numpy().reshape(1, -1), num_iteration=best_iter)[0])
        
        # 結果をリストに保存（現在値・予測値・誤差・学習に使用した決定木ラウンド数）
        results.append({
            "ticker": ticker,
            "current": round(current_price, 2),
            "predicted": round(pred_price, 2),
            "error": round(rmse, 2),
            "rounds": int(best_iter)
        })
    
        # ループ次の銘柄へ（全ての銘柄を処理）
    
    # FANG+全銘柄の合計値の変動方向を判断
    total_current = sum(item["current"] for item in results)
    total_pred = sum(item["predicted"] for item in results)
    if total_pred > total_current:
        direction = "up"
    elif total_pred < total_current:
        direction = "down"
    else:
        direction = "no_change"
    
    # JSON形式で結果を返す（各銘柄の予測結果一覧とFANG+合計の増減方向）
    return {"predictions": results, "fang_plus_direction": direction}

# アプリを実行するための設定（開発用）
# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000)
