def predict_all_stocks():
    messages = []
    total_last_close = 0
    total_pred_close = 0

    for symbol in fang_symbols:
        # 株価データ取得
        df = yf.download(symbol, period="1y", interval="1d")
        if df.empty:
            messages.append(f"{symbol}: データ取得エラー")
            continue

        # デバッグ：カラム表示
        print(f"==== {symbol} のカラム ====")
        print(df.columns)

        # カラム名修正（MultiIndex対策）
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        else:
            df.columns = [col if isinstance(col, str) else col[0] for col in df.columns]

        print(f"==== {symbol} のカラム（修正後） ====")
        print(df.columns)

        # データ前処理
        df["close_lag1"] = df["Close"].shift(1)
        df = df.dropna()

        X = df[["close_lag1"]]
        y = df["Close"]

        # 学習
        model = lgb.LGBMRegressor(n_estimators=1000)
        model.fit(X, y)

        # 予測
        last_close = df["Close"].iloc[-1]
        pred_close = model.predict([[last_close]])[0]
        trend = "↑" if pred_close > last_close else "↓"

        messages.append(f"{symbol}: {last_close:.2f} → {pred_close:.2f} {trend}")

        total_last_close += last_close
        total_pred_close += pred_close

    overall_trend = "↑" if total_pred_close > total_last_close else "↓"
    messages.append(f"\nFANG+ Index Summary:\n現終値合計: {total_last_close:.2f}\n予測終値合計: {total_pred_close:.2f}\n方向: {overall_trend}")

    return "\n".join(messages)
