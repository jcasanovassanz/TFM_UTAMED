def add_features(df):

    df = df.sort_values(["ticker", "date"])

    # LAG
    df["return_lag1"] = df.groupby("ticker")["return_1m"].shift(1)

    # MOMENTUM
    df["ret_3m"] = df.groupby("ticker")["return_1m"].rolling(3).sum().reset_index(0, drop=True)
    df["ret_6m"] = df.groupby("ticker")["return_1m"].rolling(6).sum().reset_index(0, drop=True)

    # VOLATILIDAD
    df["vol_3m"] = df.groupby("ticker")["return_1m"].rolling(3).std().reset_index(0, drop=True)
    df["vol_6m"] = df.groupby("ticker")["return_1m"].rolling(6).std().reset_index(0, drop=True)

    # EXCESO
    df["excess_return"] = df["return_1m"] - df["index_return"]

    # RANK CROSS-SECTIONAL
    df["rank_ret_3m"] = df.groupby("date")["ret_3m"].rank(pct=True)

    return df