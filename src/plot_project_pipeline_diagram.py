from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch


def draw_box(ax, x, y, w, h, title, subtitle, color):
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.03",
        linewidth=1.2,
        edgecolor="#2f2f2f",
        facecolor=color,
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h * 0.62, title, ha="center", va="center", fontsize=10.5, weight="bold")
    ax.text(x + w / 2, y + h * 0.33, subtitle, ha="center", va="center", fontsize=8.8)


def main():
    fig, ax = plt.subplots(figsize=(15, 4.8))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.axis("off")
    ax.set_xlim(0, 18)
    ax.set_ylim(0, 5)

    phases = [
        ("Data Ingestion", "Raw market + macro data", "#E8F1FA"),
        ("Feature Engineering", "Momentum, volatility,\nrelative-return features", "#E8F6EE"),
        ("Dataset Construction", "Panel dataset assembly\nwith target variable", "#FFF4DE"),
        ("Model Training", "Logistic, RF, XGBoost\nwith fixed hyperparameters", "#FDEBEC"),
        ("Walk-Forward Validation", "Expanding window:\ntrain past, test next year", "#EFEAFA"),
        ("Portfolio Simulation", "Top 20%, long-only,\nequal weight, monthly rebalance", "#E7F7F7"),
        ("Financial Evaluation", "Cumulative return, vol,\nSharpe, max drawdown", "#F3F3F3"),
    ]

    x0 = 0.4
    y = 1.5
    w = 2.2
    h = 2.0
    gap = 0.3

    for i, (title, subtitle, color) in enumerate(phases):
        x = x0 + i * (w + gap)
        draw_box(ax, x, y, w, h, title, subtitle, color)
        if i < len(phases) - 1:
            x_next = x0 + (i + 1) * (w + gap)
            ax.annotate(
                "",
                xy=(x_next - 0.03, y + h / 2),
                xytext=(x + w + 0.03, y + h / 2),
                arrowprops=dict(arrowstyle="->", lw=1.4, color="#333333"),
            )

    ax.text(
        9,
        4.45,
        "Pipeline técnico del proyecto de predicción bursátil con validación temporal",
        ha="center",
        va="center",
        fontsize=13,
    )

    out_dir = Path("results") / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "project_pipeline_diagram.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=450, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Figura guardada en: {out_path}")


if __name__ == "__main__":
    main()
