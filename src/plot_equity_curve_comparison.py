from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def build_benchmark_series(panel_path: Path) -> pd.Series:
    df = pd.read_csv(panel_path, usecols=["date", "index_return"])
    df["date"] = pd.to_datetime(df["date"])

    # index_return is repeated by ticker per month; keep one monthly observation
    benchmark = (
        df.drop_duplicates(subset=["date"])
        .sort_values("date")
        .set_index("date")["index_return"]
        .dropna()
    )
    benchmark.name = "EuroStoxx Benchmark"
    return benchmark


def main():
    base_dir = Path("results")
    figures_dir = base_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    returns_path = base_dir / "portfolio" / "portfolio_returns.csv"
    panel_path = Path("data") / "processed" / "panel_dataset_v1.csv"

    returns_df = pd.read_csv(returns_path)
    returns_df["date"] = pd.to_datetime(returns_df["date"])
    returns_df = returns_df.set_index("date").sort_index()

    benchmark = build_benchmark_series(panel_path)

    selected = pd.DataFrame(
        {
            "EuroStoxx Benchmark": benchmark,
            "Logistic Regression": returns_df["Logistic Regression | Market"],
            "Random Forest": returns_df["Random Forest | Market"],
            "XGBoost": returns_df["XGBoost | Market"],
            "XGBoost + Macro": returns_df["XGBoost | Market + Macro"],
        }
    ).dropna()

    cumulative = (1 + selected).cumprod()

    fig, ax = plt.subplots(figsize=(12.5, 6.5))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    styles = {
        "EuroStoxx Benchmark": {"color": "#222222", "linewidth": 2.2, "linestyle": "-"},
        "Logistic Regression": {"color": "#1f77b4", "linewidth": 1.9, "linestyle": "-"},
        "Random Forest": {"color": "#2ca02c", "linewidth": 1.9, "linestyle": "-"},
        "XGBoost": {"color": "#d62728", "linewidth": 1.9, "linestyle": "-"},
        "XGBoost + Macro": {"color": "#9467bd", "linewidth": 2.0, "linestyle": "--"},
    }

    for col in cumulative.columns:
        ax.plot(
            cumulative.index,
            cumulative[col],
            label=col,
            **styles[col],
        )

    ax.set_title("Evolución acumulada de estrategias de inversión", fontsize=14, pad=12)
    ax.set_xlabel("Fecha")
    ax.set_ylabel("Crecimiento acumulado (base = 1)")
    ax.grid(True, axis="both", linestyle="-", linewidth=0.6, alpha=0.30, color="#b0b0b0")

    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    ax.legend(loc="upper left", frameon=False, title="Estrategia")
    plt.tight_layout()

    out_path = figures_dir / "equity_curve_comparison.png"
    plt.savefig(out_path, dpi=400, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Figura guardada en: {out_path}")


if __name__ == "__main__":
    main()
