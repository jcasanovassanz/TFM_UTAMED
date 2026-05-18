from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier


def top_importance_df(model, feature_names, top_n=10):
    imp = pd.Series(model.feature_importances_, index=feature_names)
    imp = imp.sort_values(ascending=False).head(top_n)
    return imp.sort_values(ascending=True)


def train_models(df):
    features_market = [
        "return_lag1",
        "ret_3m",
        "ret_6m",
        "vol_3m",
        "vol_6m",
        "excess_return",
        "rank_ret_3m",
    ]
    features_macro = features_market + ["inflation", "interest_rate", "unemployment"]

    X_m = df[features_market]
    X_mm = df[features_macro]
    y = df["target"]

    rf_market = RandomForestClassifier(
        n_estimators=200,
        max_depth=5,
        min_samples_leaf=1,
        random_state=42,
    ).fit(X_m, y)
    rf_macro = RandomForestClassifier(
        n_estimators=200,
        max_depth=5,
        min_samples_leaf=1,
        random_state=42,
    ).fit(X_mm, y)

    xgb_market = XGBClassifier(
        n_estimators=50,
        max_depth=2,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric="logloss",
    ).fit(X_m, y)
    xgb_macro = XGBClassifier(
        n_estimators=50,
        max_depth=2,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric="logloss",
    ).fit(X_mm, y)

    return (
        (rf_market, features_market),
        (rf_macro, features_macro),
        (xgb_market, features_market),
        (xgb_macro, features_macro),
    )


def main():
    df = pd.read_csv("data/processed/panel_dataset_v1.csv")
    df = df.dropna()

    (rf_m, f_m), (rf_mm, f_mm), (xgb_m, _), (xgb_mm, _) = train_models(df)

    rf_m_top = top_importance_df(rf_m, f_m, top_n=10)
    rf_mm_top = top_importance_df(rf_mm, f_mm, top_n=10)
    xgb_m_top = top_importance_df(xgb_m, f_m, top_n=10)
    xgb_mm_top = top_importance_df(xgb_mm, f_mm, top_n=10)

    fig, axes = plt.subplots(2, 2, figsize=(14.5, 9.5))
    fig.patch.set_facecolor("white")

    plots = [
        (axes[0, 0], rf_m_top, "Random Forest - Market", "#1f77b4"),
        (axes[0, 1], rf_mm_top, "Random Forest - Market + Macro", "#4c9ed9"),
        (axes[1, 0], xgb_m_top, "XGBoost - Market", "#d62728"),
        (axes[1, 1], xgb_mm_top, "XGBoost - Market + Macro", "#e66b6b"),
    ]

    for ax, series, title, color in plots:
        ax.barh(series.index, series.values, color=color, edgecolor="#2f2f2f", linewidth=0.4)
        ax.set_title(title, fontsize=11, pad=8)
        ax.set_xlabel("Importancia")
        ax.grid(axis="x", linestyle="-", linewidth=0.5, alpha=0.3, color="#b0b0b0")
        ax.tick_params(axis="y", labelsize=9)
        ax.tick_params(axis="x", labelsize=9)
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)

    fig.suptitle("Importancia de variables: Random Forest y XGBoost (Top 10)", fontsize=14, y=0.98)
    plt.tight_layout(rect=(0, 0, 1, 0.96))

    out_dir = Path("results") / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "feature_importance_rf_xgb.png"
    plt.savefig(out_path, dpi=450, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Figura guardada en: {out_path}")


if __name__ == "__main__":
    main()
