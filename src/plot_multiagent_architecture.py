from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch


def add_agent(ax, x, y, w, h, title, subtitle, color):
    box = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.03,rounding_size=0.04",
        linewidth=1.2,
        edgecolor="#2e2e2e",
        facecolor=color,
    )
    ax.add_patch(box)
    ax.text(x + w / 2, y + h * 0.62, title, ha="center", va="center", fontsize=10.2, weight="bold")
    ax.text(x + w / 2, y + h * 0.34, subtitle, ha="center", va="center", fontsize=8.5)


def arrow(ax, x0, y0, x1, y1, text=None):
    ax.annotate(
        "",
        xy=(x1, y1),
        xytext=(x0, y0),
        arrowprops=dict(arrowstyle="->", lw=1.3, color="#333333"),
    )
    if text:
        ax.text((x0 + x1) / 2, (y0 + y1) / 2 + 0.15, text, ha="center", va="center", fontsize=8, color="#404040")


def main():
    fig, ax = plt.subplots(figsize=(14.8, 8.4))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.axis("off")
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 10)

    # Orchestrator at top center
    add_agent(
        ax,
        6.1,
        8.1,
        3.8,
        1.35,
        "Orchestrator Agent",
        "Coordina tareas, dependencias\ny ciclo de ejecución",
        "#EDE9FE",
    )

    # Middle layer
    add_agent(
        ax,
        0.8,
        5.7,
        3.2,
        1.45,
        "Data Agent",
        "Ingesta y validación de\nseries de mercado",
        "#E7F1FA",
    )
    add_agent(
        ax,
        4.5,
        5.7,
        3.2,
        1.45,
        "Feature Engineering Agent",
        "Construcción de señales\ny variables predictivas",
        "#E8F6EE",
    )
    add_agent(
        ax,
        8.2,
        5.7,
        3.2,
        1.45,
        "Macro Scenario Agent",
        "Integración de factores macro\ny escenarios",
        "#FFF4DE",
    )
    add_agent(
        ax,
        11.9,
        5.7,
        3.2,
        1.45,
        "Model Agent",
        "Entrenamiento + validación\nwalk-forward",
        "#FDEBEC",
    )

    # Bottom layer
    add_agent(
        ax,
        4.2,
        2.6,
        3.8,
        1.55,
        "Portfolio Simulation Agent",
        "Selección top 20%, long-only,\nequal-weight, rebalanceo mensual",
        "#E7F7F7",
    )
    add_agent(
        ax,
        8.6,
        2.6,
        3.8,
        1.55,
        "Evaluation Agent",
        "Métricas financieras,\nequity curve y drawdown",
        "#F3F3F3",
    )

    # Orchestrator control arrows
    arrow(ax, 8.0, 8.1, 2.4, 7.2, "Planificación")
    arrow(ax, 8.0, 8.1, 6.1, 7.2, "Control")
    arrow(ax, 8.0, 8.1, 9.8, 7.2, "Control")
    arrow(ax, 8.0, 8.1, 13.5, 7.2, "Control")
    arrow(ax, 8.0, 8.1, 6.1, 4.2, "Orquestación")
    arrow(ax, 8.0, 8.1, 10.5, 4.2, "Orquestación")

    # Data flow arrows
    arrow(ax, 4.0, 6.45, 4.5, 6.45, "Datos limpios")
    arrow(ax, 4.0, 6.1, 8.2, 6.1, "Dataset base")
    arrow(ax, 7.7, 6.45, 11.9, 6.45, "Features")
    arrow(ax, 11.4, 6.1, 11.9, 6.1, "Variables macro")
    arrow(ax, 13.5, 5.7, 6.1, 4.15, "Probabilidades")
    arrow(ax, 8.0, 3.35, 8.6, 3.35, "Retornos de cartera")

    ax.text(
        8.0,
        9.75,
        "Arquitectura conceptual multiagente para análisis financiero",
        ha="center",
        va="center",
        fontsize=14,
    )

    out_dir = Path("results") / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "multiagent_financial_architecture.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=450, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Figura guardada en: {out_path}")


if __name__ == "__main__":
    main()
