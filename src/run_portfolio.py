import pandas as pd

from portfolio import simulate_portfolio, get_logistic, get_rf, get_xgb
from metrics import portfolio_metrics

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

log_metrics = portfolio_metrics(log_returns)
rf_metrics = portfolio_metrics(rf_returns)
xgb_metrics = portfolio_metrics(xgb_returns)

print("\nLogistic Portfolio Metrics")
for k, v in log_metrics.items():
    print(f"{k}: {v}")

print("\nRandom Forest Portfolio Metrics")
for k, v in rf_metrics.items():
    print(f"{k}: {v}")

print("\nXGBoost Portfolio Metrics")
for k, v in xgb_metrics.items():
    print(f"{k}: {v}")

print("\nRetornos cartera (ejemplo):")
print(log_returns.head())