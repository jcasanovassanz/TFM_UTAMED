from pathlib import Path

import matplotlib.pyplot as plt


def main():
    start_train_year = 2010
    start_test_year = 2015
    end_test_year = 2024

    rows = []
    for test_year in range(start_test_year, end_test_year + 1):
        rows.append(
            {
                "train_start": start_train_year,
                "train_end": test_year - 1,
                "test_year": test_year,
            }
        )

    fig, ax = plt.subplots(figsize=(16, 6.5))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    train_color = "#4C78A8"
    test_color = "#E45756"

    for i, row in enumerate(rows):
        y_pos = len(rows) - 1 - i
        train_width = row["train_end"] - row["train_start"] + 1

        ax.barh(
            y=y_pos,
            width=train_width,
            left=row["train_start"],
            height=0.6,
            color=train_color,
            edgecolor="black",
            linewidth=0.5,
            label="Entrenamiento" if i == 0 else None,
        )
        ax.barh(
            y=y_pos,
            width=1,
            left=row["test_year"],
            height=0.6,
            color=test_color,
            edgecolor="black",
            linewidth=0.5,
            label="Test" if i == 0 else None,
        )

        label = f"Train {row['train_start']}-{row['train_end']} -> Test {row['test_year']}"
        ax.text(2009.05, y_pos, label, va="center", ha="right", fontsize=9)

    ax.set_xlim(2008.2, 2025.8)
    ax.set_xticks(range(2010, 2025))
    ax.set_yticks([])
    ax.set_xlabel("Anio")
    ax.set_title("Esquema de validacion temporal walk-forward", fontsize=14, pad=12)
    ax.grid(axis="x", color="#d9d9d9", linestyle="-", linewidth=0.6, alpha=0.7)
    ax.legend(loc="upper right", frameon=False)

    for spine in ["top", "right", "left"]:
        ax.spines[spine].set_visible(False)

    output_dir = Path("results") / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "walk_forward_validation.png"

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    print(f"Figura guardada en: {output_path}")


if __name__ == "__main__":
    main()
