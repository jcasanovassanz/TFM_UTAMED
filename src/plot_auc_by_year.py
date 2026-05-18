from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def load_auc(path):
    df = pd.read_csv(path)
    return df[["test_year", "auc"]].copy()


def main():
    base = Path("results")
    figures_dir = base / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    series = {
        "Logistic Regression | Market": load_auc(base / "logistic_market_by_year.csv"),
        "Logistic Regression | Market + Macro": load_auc(base / "logistic_macro_by_year.csv"),
        "Random Forest | Market": load_auc(base / "rf_market_by_year.csv"),
        "Random Forest | Market + Macro": load_auc(base / "rf_macro_by_year.csv"),
        "XGBoost | Market": load_auc(base / "xgb_market_by_year.csv"),
        "XGBoost | Market + Macro": load_auc(base / "xgb_macro_by_year.csv"),
    }

    colors = {
        "Logistic Regression": "#1f77b4",
        "Random Forest": "#2ca02c",
        "XGBoost": "#d62728",
    }

    fig, ax = plt.subplots(figsize=(12, 6.5))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    for label, df in series.items():
        model = label.split(" | ")[0]
        is_macro = "Macro" in label
        ax.plot(
            df["test_year"],
            df["auc"],
            label=label,
            color=colors[model],
            linewidth=1.8,
            linestyle="--" if is_macro else "-",
            marker="o" if is_macro else "s",
            markersize=4.5,
        )

    ax.set_title("Evolucion anual del AUC por modelo y conjunto de variables", fontsize=14, pad=12)
    ax.set_xlabel("Anio de test")
    ax.set_ylabel("AUC")
    ax.set_xticks(sorted(series["Logistic Regression | Market"]["test_year"].unique()))
    ax.set_ylim(0.40, 0.63)
    ax.grid(True, axis="both", linestyle="-", linewidth=0.6, alpha=0.35, color="#b0b0b0")

    ax.legend(
        title="Modelo y configuracion",
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        frameon=False,
    )

    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    out_path = figures_dir / "auc_evolution_by_year.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=400, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Figura guardada en: {out_path}")


if __name__ == "__main__":
    main()
