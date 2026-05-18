from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def prepare_matrix(df, feature_set):
    cols = ["accuracy", "precision", "recall", "f1", "auc"]
    subset = df[df["feature_set"] == feature_set].copy()
    subset = subset.set_index("model")[cols]
    return subset


def main():
    results_dir = Path("results")
    figures_dir = results_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    summary_path = results_dir / "model_comparison_summary.csv"
    df = pd.read_csv(summary_path)

    market = prepare_matrix(df, "Market")
    macro = prepare_matrix(df, "Market + Macro")

    sns.set_theme(style="white")
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.5), sharey=True)
    fig.patch.set_facecolor("white")

    cmap = sns.color_palette("Blues", as_cmap=True)

    sns.heatmap(
        market,
        ax=axes[0],
        annot=True,
        fmt=".4f",
        cmap=cmap,
        cbar=False,
        linewidths=0.5,
        linecolor="white",
        vmin=0.45,
        vmax=1.00,
    )
    axes[0].set_title("Market", fontsize=12, pad=10)
    axes[0].set_xlabel("Metrica")
    axes[0].set_ylabel("Modelo")

    sns.heatmap(
        macro,
        ax=axes[1],
        annot=True,
        fmt=".4f",
        cmap=cmap,
        cbar=True,
        cbar_kws={"shrink": 0.9, "label": "Valor"},
        linewidths=0.5,
        linecolor="white",
        vmin=0.45,
        vmax=1.00,
    )
    axes[1].set_title("Market + Macro", fontsize=12, pad=10)
    axes[1].set_xlabel("Metrica")
    axes[1].set_ylabel("")

    for ax in axes:
        ax.tick_params(axis="x", rotation=25)
        ax.tick_params(axis="y", rotation=0)

    fig.suptitle("Comparacion de metricas por modelo", fontsize=14, y=1.02)
    plt.tight_layout()

    out_path = figures_dir / "model_metrics_heatmap.png"
    plt.savefig(out_path, dpi=400, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Figura guardada en: {out_path}")


if __name__ == "__main__":
    main()
