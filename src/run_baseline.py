import os
import pandas as pd

from model import run_walk_forward_validation
from model import run_walk_forward_validation_rf
from model import run_walk_forward_validation_xgb

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
# LOGISTIC REGRESSION
# -----------------------

results_log_market = run_walk_forward_validation(
    df,
    features_market,
    start_test_year=2015
)

results_log_macro = run_walk_forward_validation(
    df,
    features_macro,
    start_test_year=2015
)

# -----------------------
# RANDOM FOREST (TUNED)
# Best params from tuning:
# n_estimators=200, max_depth=5, min_samples_leaf=1
# -----------------------

results_rf_market = run_walk_forward_validation_rf(
    df,
    features_market,
    start_test_year=2015,
    n_estimators=200,
    max_depth=5,
    min_samples_leaf=1
)

results_rf_macro = run_walk_forward_validation_rf(
    df,
    features_macro,
    start_test_year=2015,
    n_estimators=200,
    max_depth=5,
    min_samples_leaf=1
)

# -----------------------
# XGBOOST (TUNED)
# Best params from tuning:
# n_estimators=50, max_depth=2, learning_rate=0.03
# -----------------------

results_xgb_market = run_walk_forward_validation_xgb(
    df,
    features_market,
    start_test_year=2015,
    n_estimators=50,
    max_depth=2,
    learning_rate=0.03
)

results_xgb_macro = run_walk_forward_validation_xgb(
    df,
    features_macro,
    start_test_year=2015,
    n_estimators=50,
    max_depth=2,
    learning_rate=0.03
)

# -----------------------
# FUNCIÓN RESUMEN
# -----------------------

def summarize_results(model_name, feature_set, results_df):
    return {
        "model": model_name,
        "feature_set": feature_set,
        "accuracy": results_df["accuracy"].mean(),
        "precision": results_df["precision"].mean(),
        "recall": results_df["recall"].mean(),
        "f1": results_df["f1"].mean(),
        "auc": results_df["auc"].mean()
    }

# -----------------------
# TABLA FINAL COMPARATIVA
# -----------------------

comparison = pd.DataFrame([
    summarize_results("Logistic Regression", "Market", results_log_market),
    summarize_results("Logistic Regression", "Market + Macro", results_log_macro),
    summarize_results("Random Forest", "Market", results_rf_market),
    summarize_results("Random Forest", "Market + Macro", results_rf_macro),
    summarize_results("XGBoost", "Market", results_xgb_market),
    summarize_results("XGBoost", "Market + Macro", results_xgb_macro),
])

comparison = comparison.sort_values(["model", "feature_set"]).reset_index(drop=True)

# -----------------------
# MOSTRAR RESULTADOS
# -----------------------

print("\nComparativa final de modelos:")
print(comparison)

# -----------------------
# GUARDAR RESULTADOS
# -----------------------

os.makedirs("results", exist_ok=True)

comparison.to_csv("results/model_comparison_summary.csv", index=False)
results_log_market.to_csv("results/logistic_market_by_year.csv", index=False)
results_log_macro.to_csv("results/logistic_macro_by_year.csv", index=False)
results_rf_market.to_csv("results/rf_market_by_year.csv", index=False)
results_rf_macro.to_csv("results/rf_macro_by_year.csv", index=False)
results_xgb_market.to_csv("results/xgb_market_by_year.csv", index=False)
results_xgb_macro.to_csv("results/xgb_macro_by_year.csv", index=False)

print("\nArchivos guardados en la carpeta /results")