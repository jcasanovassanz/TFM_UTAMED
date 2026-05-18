from pathlib import Path

import pandas as pd


def build_main_table(metrics_path: Path) -> pd.DataFrame:
    df = pd.read_csv(metrics_path)
    out = df[
        [
            "model",
            "feature_set",
            "cumulative_return",
            "annualized_volatility",
            "sharpe_ratio",
            "max_drawdown",
        ]
    ].copy()
    metric_cols = [
        "cumulative_return",
        "annualized_volatility",
        "sharpe_ratio",
        "max_drawdown",
    ]
    out[metric_cols] = out[metric_cols].round(4)
    out = out.sort_values(["feature_set", "model"]).reset_index(drop=True)
    return out


def build_optional_table(returns_path: Path) -> pd.DataFrame:
    returns_df = pd.read_csv(returns_path)
    strategy_cols = [c for c in returns_df.columns if c != "date"]

    rows = []
    for col in strategy_cols:
        model, feature_set = [x.strip() for x in col.split("|")]
        s = returns_df[col].dropna()
        rows.append(
            {
                "model": model,
                "feature_set": feature_set,
                "monthly_return_mean": s.mean(),
                "monthly_return_std": s.std(ddof=1),
                "best_month": s.max(),
                "worst_month": s.min(),
                "number_of_rebalances": int(s.shape[0]),
            }
        )

    out = pd.DataFrame(rows)
    metric_cols = ["monthly_return_mean", "monthly_return_std", "best_month", "worst_month"]
    out[metric_cols] = out[metric_cols].round(4)
    out = out.sort_values(["feature_set", "model"]).reset_index(drop=True)
    return out


def main():
    results_dir = Path("results")
    portfolio_dir = results_dir / "portfolio"

    main_table = build_main_table(portfolio_dir / "portfolio_metrics.csv")
    optional_table = build_optional_table(portfolio_dir / "portfolio_returns.csv")

    out_csv = results_dir / "anexo_d_portfolio_metrics.csv"
    out_xlsx = results_dir / "anexo_d_portfolio_metrics.xlsx"
    out_opt_csv = results_dir / "anexo_d_portfolio_metrics_optional.csv"

    main_table.to_csv(out_csv, index=False)
    optional_table.to_csv(out_opt_csv, index=False)

    with pd.ExcelWriter(out_xlsx) as writer:
        main_table.to_excel(writer, sheet_name="anexo_d_main", index=False)
        optional_table.to_excel(writer, sheet_name="anexo_d_optional", index=False)

    print(f"Archivo principal CSV generado: {out_csv}")
    print(f"Archivo principal Excel generado: {out_xlsx}")
    print(f"Archivo opcional CSV generado: {out_opt_csv}")
    print(f"Filas tabla principal: {len(main_table)}")
    print(f"Filas tabla opcional: {len(optional_table)}")


if __name__ == "__main__":
    main()
