import pandas as pd

from model import run_walk_forward_validation

# -----------------------
# CARGA DATASET
# -----------------------

df = pd.read_csv("data/processed/panel_dataset_v1.csv")
df["date"] = pd.to_datetime(df["date"])

# -----------------------
# FEATURES
# -----------------------

features_market = [
    "return_lag1",
    "ret_3m",
    "ret_6m",
    "vol_3m",
    "vol_6m",
    "excess_return",
    "rank_ret_3m"
]

features_macro = features_market + [
    "inflation",
    "interest_rate",
    "unemployment"
]

# -----------------------
# VALIDACIÓN
# -----------------------

results_market = run_walk_forward_validation(
    df,
    features_market,
    start_test_year=2015
)

results_macro = run_walk_forward_validation(
    df,
    features_macro,
    start_test_year=2015
)

# -----------------------
# RESULTADOS
# -----------------------

print("\nResultados modelo solo mercado:")
print(results_market)

print("\nResultados modelo mercado + macro:")
print(results_macro)

print("\nMedia modelo solo mercado:")
print(results_market.mean(numeric_only=True))

print("\nMedia modelo mercado + macro:")
print(results_macro.mean(numeric_only=True))