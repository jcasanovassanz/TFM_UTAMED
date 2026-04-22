import pandas as pd

from portfolio import simulate_portfolio, get_logistic, get_rf, get_xgb

# cargar datos
df = pd.read_csv("data/processed/panel_dataset_v1.csv")
df["date"] = pd.to_datetime(df["date"])

features_market = [
    "return_lag1",
    "ret_3m",
    "ret_6m",
    "vol_3m",
    "vol_6m",
    "excess_return",
    "rank_ret_3m"
]

# simular
log_returns = simulate_portfolio(df, get_logistic, features_market)
rf_returns = simulate_portfolio(df, get_rf, features_market)
xgb_returns = simulate_portfolio(df, get_xgb, features_market)

print("\nRetornos cartera (ejemplo):")
print(log_returns.head())