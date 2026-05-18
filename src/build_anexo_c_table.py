from pathlib import Path

import pandas as pd


def load_metrics(path, model, feature_set):
    df = pd.read_csv(path)
    out = df[["test_year", "accuracy", "precision", "recall", "f1", "auc"]].copy()
    out = out.rename(columns={"test_year": "year"})
    out["model"] = model
    out["feature_set"] = feature_set
    return out


def main():
    results_dir = Path("results")

    frames = [
        load_metrics(results_dir / "logistic_market_by_year.csv", "Logistic Regression", "Market"),
        load_metrics(results_dir / "logistic_macro_by_year.csv", "Logistic Regression", "Market + Macro"),
        load_metrics(results_dir / "rf_market_by_year.csv", "Random Forest", "Market"),
        load_metrics(results_dir / "rf_macro_by_year.csv", "Random Forest", "Market + Macro"),
        load_metrics(results_dir / "xgb_market_by_year.csv", "XGBoost", "Market"),
        load_metrics(results_dir / "xgb_macro_by_year.csv", "XGBoost", "Market + Macro"),
    ]

    table = pd.concat(frames, ignore_index=True)
    table = table[
        ["year", "model", "feature_set", "accuracy", "precision", "recall", "f1", "auc"]
    ].copy()

    metric_cols = ["accuracy", "precision", "recall", "f1", "auc"]
    table[metric_cols] = table[metric_cols].round(4)

    table = table.sort_values(["year", "feature_set", "model"]).reset_index(drop=True)

    csv_path = results_dir / "anexo_c_walkforward_metrics.csv"
    xlsx_path = results_dir / "anexo_c_walkforward_metrics.xlsx"

    table.to_csv(csv_path, index=False)
    table.to_excel(xlsx_path, index=False)

    print(f"Archivo CSV generado: {csv_path}")
    print(f"Archivo Excel generado: {xlsx_path}")
    print(f"Filas totales: {len(table)}")


if __name__ == "__main__":
    main()
