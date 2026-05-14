from pathlib import Path
from datetime import datetime, timezone

import pandas as pd

from metrics import portfolio_metrics
from portfolio import get_logistic, get_rf, get_xgb, simulate_portfolio


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "processed" / "panel_dataset_v1.csv"
RESULTS_DIR = BASE_DIR / "results" / "portfolio"


FEATURES_MARKET = [
    "return_lag1",
    "ret_3m",
    "ret_6m",
    "vol_3m",
    "vol_6m",
    "excess_return",
    "rank_ret_3m",
]

FEATURES_MACRO = FEATURES_MARKET + [
    "inflation",
    "interest_rate",
    "unemployment",
]


def build_strategies():
    return [
        ("Logistic Regression", "Market", get_logistic, FEATURES_MARKET),
        ("Logistic Regression", "Market + Macro", get_logistic, FEATURES_MACRO),
        ("Random Forest", "Market", get_rf, FEATURES_MARKET),
        ("Random Forest", "Market + Macro", get_rf, FEATURES_MACRO),
        ("XGBoost", "Market", get_xgb, FEATURES_MARKET),
        ("XGBoost", "Market + Macro", get_xgb, FEATURES_MACRO),
    ]


def main():
    df = pd.read_csv(DATA_PATH)
    df["date"] = pd.to_datetime(df["date"])

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    metrics_rows = []
    returns_series = {}

    for model_name, feature_set, model_func, features in build_strategies():
        portfolio_returns = simulate_portfolio(
            df,
            model_func,
            features,
            top_pct=0.2,
            start_test_year=2015,
        )

        strategy_name = f"{model_name} | {feature_set}"
        returns_series[strategy_name] = portfolio_returns

        metrics = portfolio_metrics(portfolio_returns)
        metrics_rows.append(
            {
                "model": model_name,
                "feature_set": feature_set,
                "cumulative_return": metrics["cumulative_return"],
                "annualized_return": metrics["annualized_return"],
                "annualized_volatility": metrics["annualized_volatility"],
                "sharpe_ratio": metrics["sharpe_ratio"],
                "max_drawdown": metrics["max_drawdown"],
            }
        )

    metrics_df = pd.DataFrame(metrics_rows)
    metrics_df = metrics_df.sort_values(["model", "feature_set"]).reset_index(drop=True)

    returns_df = pd.concat(returns_series, axis=1).sort_index()
    cumulative_df = (1 + returns_df).cumprod()

    metrics_path = RESULTS_DIR / "portfolio_metrics.csv"
    returns_path = RESULTS_DIR / "portfolio_returns.csv"
    cumulative_path = RESULTS_DIR / "cumulative_returns.csv"

    metrics_df.to_csv(metrics_path, index=False)
    returns_df.to_csv(returns_path, index_label="date")
    cumulative_df.to_csv(cumulative_path, index_label="date")

    run_info = pd.DataFrame(
        [
            {
                "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                "script": "src/run_portfolio.py",
                "dataset_path": str(DATA_PATH.relative_to(BASE_DIR)).replace("\\", "/"),
                "n_rows": len(df),
                "n_columns": len(df.columns),
                "start_test_year": 2015,
                "top_pct": 0.2,
                "n_configurations": len(build_strategies()),
            }
        ]
    )
    run_info.to_csv(RESULTS_DIR / "run_info.csv", index=False)

    print("\nPortfolio metrics summary:")
    print(metrics_df.to_string(index=False))


if __name__ == "__main__":
    main()
