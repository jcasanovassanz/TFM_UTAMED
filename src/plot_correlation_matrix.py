from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def main():
    df = pd.read_csv("data/processed/panel_dataset_v1.csv")

    cols = [
        "return_1m",
        "return_lag1",
        "ret_3m",
        "ret_6m",
        "vol_3m",
        "vol_6m",
        "excess_return",
        "rank_ret_3m",
        "index_return",
        "inflation",
        "interest_rate",
        "unemployment",
    ]

    corr = df[cols].corr(numeric_only=True)

    sns.set_theme(style="white")
    fig, ax = plt.subplots(figsize=(11.5, 9.2))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    sns.heatmap(
        corr,
        ax=ax,
        cmap="RdBu_r",
        vmin=-1,
        vmax=1,
        center=0,
        annot=True,
        fmt=".2f",
        linewidths=0.5,
        linecolor="white",
        cbar_kws={"label": "Correlación"},
        square=True,
    )

    ax.set_title("Matriz de correlación de variables financieras y macroeconómicas", fontsize=13, pad=12)
    ax.tick_params(axis="x", rotation=35, labelsize=9)
    ax.tick_params(axis="y", rotation=0, labelsize=9)

    plt.tight_layout()

    out_dir = Path("results") / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "correlation_matrix_main_variables.png"
    plt.savefig(out_path, dpi=450, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    print(f"Figura guardada en: {out_path}")


if __name__ == "__main__":
    main()
