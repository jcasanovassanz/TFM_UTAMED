import matplotlib

matplotlib.use("Agg")

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

try:
    import shap
except ImportError as exc:
    raise ImportError(
        "La librería 'shap' no está instalada. Instálala antes de ejecutar este script."
    ) from exc

from xgboost import XGBClassifier


ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = ROOT / "data" / "processed" / "panel_dataset_v1.csv"
OUTPUT_DIR = ROOT / "results" / "shap"


# Configuración exacta usada en el pipeline principal para XGBoost.
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

XGB_PARAMS = {
    "n_estimators": 50,
    "max_depth": 2,
    "learning_rate": 0.03,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": 42,
    "eval_metric": "logloss",
}


def load_dataset() -> pd.DataFrame:
    """Carga el dataset panel final ya construido por el pipeline existente."""
    df = pd.read_csv(DATASET_PATH)
    df["date"] = pd.to_datetime(df["date"])
    return df


def split_train_test_last_year(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    """
    Reproduce la lógica temporal walk-forward del proyecto.
    Para el análisis SHAP usamos el último año disponible como test.
    """
    df = df.copy()
    df["year"] = df["date"].dt.year
    last_year = int(df["year"].max())

    train = df[df["year"] < last_year].copy()
    test = df[df["year"] == last_year].copy()

    if train.empty or test.empty:
        raise ValueError("No se pudo crear una partición train/test válida con el último año disponible.")

    return train, test, last_year


def train_xgb(train: pd.DataFrame, features: list[str]) -> XGBClassifier:
    """Entrena XGBoost con los hiperparámetros reales del proyecto."""
    model = XGBClassifier(**XGB_PARAMS)
    model.fit(train[features], train["target"])
    return model


def save_feature_importance_csv(shap_df: pd.DataFrame, output_path: Path) -> pd.DataFrame:
    """Guarda la importancia media absoluta de SHAP por variable."""
    importance = shap_df.abs().mean().sort_values(ascending=False).reset_index()
    importance.columns = ["feature", "mean_abs_shap"]
    importance.to_csv(output_path, index=False)
    return importance


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = load_dataset()
    train, test, last_year = split_train_test_last_year(df)

    # Usamos únicamente Market + Macro, como pediste.
    features = FEATURES_MACRO

    model = train_xgb(train, features)

    # SHAP sobre el conjunto de test temporalmente más reciente.
    X_test = test[features]

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    # SHAP puede devolver lista en algunas versiones; aquí normalizamos al caso binario.
    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    shap_df = pd.DataFrame(shap_values, columns=features, index=X_test.index)

    summary_path = OUTPUT_DIR / "shap_summary_xgb_macro.png"
    bar_path = OUTPUT_DIR / "shap_importance_xgb_macro.png"
    csv_path = OUTPUT_DIR / "shap_importance_xgb_macro.csv"

    # Beeswarm: muestra dirección y dispersión del efecto por observación.
    plt.figure()
    shap.summary_plot(shap_values, X_test, show=False)
    plt.savefig(summary_path, dpi=300, bbox_inches="tight")
    plt.close()

    # Bar plot: importancia global media de cada variable.
    plt.figure()
    shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
    plt.savefig(bar_path, dpi=300, bbox_inches="tight")
    plt.close()

    importance_df = save_feature_importance_csv(shap_df, csv_path)

    run_info = pd.DataFrame(
        [
            {
                "dataset_path": str(DATASET_PATH),
                "train_years": f"{int(train['year'].min())}-{int(train['year'].max())}",
                "test_year": last_year,
                "n_train": len(train),
                "n_test": len(test),
                "model": "XGBoost",
                "feature_set": "Market + Macro",
                "n_features": len(features),
                "summary_path": str(summary_path),
                "bar_path": str(bar_path),
                "csv_path": str(csv_path),
            }
        ]
    )
    run_info.to_csv(OUTPUT_DIR / "shap_run_info.csv", index=False)

    print("SHAP analysis completed.")
    print(f"Test year: {last_year}")
    print(f"Summary plot: {summary_path}")
    print(f"Bar plot: {bar_path}")
    print(f"Importance CSV: {csv_path}")
    print("\nTop features by mean |SHAP|:")
    print(importance_df.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
