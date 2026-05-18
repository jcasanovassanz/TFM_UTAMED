from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch


def add_box(ax, x, y, w, h, text, fc, ec="#333333", fs=10):
    box = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.02",
        linewidth=1.0,
        edgecolor=ec,
        facecolor=fc,
    )
    ax.add_patch(box)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs)


def main():
    fig, ax = plt.subplots(figsize=(14, 5.5))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.axis("off")

    ax.set_xlim(0, 15)
    ax.set_ylim(0, 6)

    # Main timeline blocks
    add_box(
        ax,
        0.7,
        3.6,
        2.5,
        1.2,
        "Variables explicativas\nobservadas en t\n(features de mercado y macro)",
        fc="#DCEBFA",
    )
    add_box(
        ax,
        4.0,
        3.6,
        2.2,
        1.2,
        "Entrenamiento del modelo\ncon datos historicos\nhasta t",
        fc="#E6F4E8",
    )
    add_box(
        ax,
        7.0,
        3.6,
        2.0,
        1.2,
        "Prediccion en t\nP(y=1 | X_t)",
        fc="#FFF3D6",
    )
    add_box(
        ax,
        9.8,
        3.6,
        2.7,
        1.2,
        "Retorno relativo\nobservado en t+1\n(r_i,t+1 - r_m,t+1)",
        fc="#FDE3E3",
    )
    add_box(
        ax,
        13.0,
        3.6,
        1.3,
        1.2,
        "Target\nbinario y_t",
        fc="#E8E0FA",
    )

    # Arrows
    arrow = dict(arrowstyle="->", lw=1.3, color="#333333")
    ax.annotate("", xy=(4.0, 4.2), xytext=(3.2, 4.2), arrowprops=arrow)
    ax.annotate("", xy=(7.0, 4.2), xytext=(6.2, 4.2), arrowprops=arrow)
    ax.annotate("", xy=(9.8, 4.2), xytext=(9.0, 4.2), arrowprops=arrow)
    ax.annotate("", xy=(13.0, 4.2), xytext=(12.5, 4.2), arrowprops=arrow)

    # Rule for binary target
    add_box(
        ax,
        9.0,
        1.2,
        5.0,
        1.3,
        "Regla de construccion del target:\n"
        "y_t = 1 si (r_i,t+1 - r_m,t+1) > 0;  y_t = 0 en otro caso",
        fc="#F7F7F7",
        fs=10,
    )
    ax.annotate("", xy=(13.65, 3.6), xytext=(11.5, 2.5), arrowprops=arrow)

    # Time markers
    ax.text(1.9, 5.15, "Tiempo t", ha="center", fontsize=10)
    ax.text(11.1, 5.15, "Tiempo t+1", ha="center", fontsize=10)
    ax.plot([1.2, 8.7], [5.0, 5.0], color="#666666", linewidth=0.9)
    ax.plot([9.9, 14.1], [5.0, 5.0], color="#666666", linewidth=0.9)

    ax.set_title(
        "Construccion temporal de la variable objetivo (target) del modelo",
        fontsize=14,
        pad=12,
    )

    out_dir = Path("results") / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "target_construction_timeline.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=400, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Figura guardada en: {out_path}")


if __name__ == "__main__":
    main()
