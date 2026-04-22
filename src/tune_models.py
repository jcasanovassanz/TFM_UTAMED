import pandas as pd

from model import run_walk_forward_validation_rf, run_walk_forward_validation_xgb

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
# RANDOM FOREST TUNING
# -----------------------

rf_grid = [
    {"n_estimators": 100, "max_depth": 3, "min_samples_leaf": 1},
    {"n_estimators": 100, "max_depth": 5, "min_samples_leaf": 1},
    {"n_estimators": 200, "max_depth": 5, "min_samples_leaf": 1},
    {"n_estimators": 200, "max_depth": 8, "min_samples_leaf": 5},
]

rf_results = []

for params in rf_grid:
    result = run_walk_forward_validation_rf(
        df,
        features_market,
        start_test_year=2015,
        n_estimators=params["n_estimators"],
        max_depth=params["max_depth"],
        min_samples_leaf=params["min_samples_leaf"]
    )

    rf_results.append({
        "model": "RandomForest",
        "features": "market",
        **params,
        "mean_auc": result["auc"].mean(),
        "mean_f1": result["f1"].mean(),
        "mean_accuracy": result["accuracy"].mean()
    })

# -----------------------
# XGBOOST TUNING
# -----------------------

xgb_grid = [
    {"n_estimators": 50, "max_depth": 2, "learning_rate": 0.03},
    {"n_estimators": 100, "max_depth": 3, "learning_rate": 0.05},
    {"n_estimators": 100, "max_depth": 4, "learning_rate": 0.05},
    {"n_estimators": 200, "max_depth": 3, "learning_rate": 0.1},
]

xgb_results = []

for params in xgb_grid:
    result = run_walk_forward_validation_xgb(
        df,
        features_market,
        start_test_year=2015,
        n_estimators=params["n_estimators"],
        max_depth=params["max_depth"],
        learning_rate=params["learning_rate"]
    )

    xgb_results.append({
        "model": "XGBoost",
        "features": "market",
        **params,
        "mean_auc": result["auc"].mean(),
        "mean_f1": result["f1"].mean(),
        "mean_accuracy": result["accuracy"].mean()
    })

# -----------------------
# MOSTRAR RESULTADOS
# -----------------------

rf_df = pd.DataFrame(rf_results).sort_values("mean_auc", ascending=False)
xgb_df = pd.DataFrame(xgb_results).sort_values("mean_auc", ascending=False)

print("\nRandom Forest tuning:")
print(rf_df)

print("\nXGBoost tuning:")
print(xgb_df)