def add_features(df):

    df = df.sort_values(["ticker", "date"]).copy()

    # LAG
    df["return_lag1"] = df.groupby("ticker")["return_1m"].shift(1)

    # MOMENTUM usando solo información pasada
    df["ret_3m"] = (
        df.groupby("ticker")["return_lag1"]
        .rolling(3)
        .sum()
        .reset_index(level=0, drop=True)
    )

    df["ret_6m"] = (
        df.groupby("ticker")["return_lag1"]
        .rolling(6)
        .sum()
        .reset_index(level=0, drop=True)
    )

    # VOLATILIDAD usando solo información pasada
    df["vol_3m"] = (
        df.groupby("ticker")["return_lag1"]
        .rolling(3)
        .std()
        .reset_index(level=0, drop=True)
    )

    df["vol_6m"] = (
        df.groupby("ticker")["return_lag1"]
        .rolling(6)
        .std()
        .reset_index(level=0, drop=True)
    )

    # EXCESO DE RETORNO CONTEMPORÁNEO
    df["excess_return"] = df["return_1m"] - df["index_return"]

    # RANKING CROSS-SECTIONAL
    df["rank_ret_3m"] = df.groupby("date")["ret_3m"].rank(pct=True)

    return df