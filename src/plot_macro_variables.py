from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]

possible_files = [
    ROOT / "data" / "processed" / "macro_real.csv",
    ROOT / "data" / "macro" / "macro.csv",
    ROOT / "data" / "processed" / "panel_dataset_v1.csv",
]

input_file = next((p for p in possible_files if p.exists()), None)

if input_file is None:
    raise FileNotFoundError("No se ha encontrado ningún archivo macroeconómico válido.")

df = pd.read_csv(input_file)

date_candidates = ["date", "Date", "fecha", "Fecha"]
date_col = next((c for c in date_candidates if c in df.columns), None)

if date_col is None:
    raise ValueError(f"No se encontró columna de fecha. Columnas disponibles: {df.columns.tolist()}")

required_cols = ["inflation", "interest_rate", "unemployment"]
missing = [c for c in required_cols if c not in df.columns]

if missing:
    raise ValueError(f"Faltan columnas macroeconómicas: {missing}. Columnas disponibles: {df.columns.tolist()}")

df[date_col] = pd.to_datetime(df[date_col])
df = df.sort_values(date_col)

# Si viene del panel dataset, puede haber duplicados por ticker/fecha.
df = df[[date_col] + required_cols].drop_duplicates(subset=[date_col])

output_dir = ROOT / "results" / "figures"
output_dir.mkdir(parents=True, exist_ok=True)

fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True)

series_info = [
    ("inflation", "Inflación"),
    ("interest_rate", "Tipo de interés"),
    ("unemployment", "Tasa de desempleo"),
]

for ax, (col, label) in zip(axes, series_info):
    ax.plot(df[date_col], df[col], linewidth=1.8)
    ax.set_ylabel(label)
    ax.grid(True, alpha=0.3)

fig.suptitle("Evolución de las variables macroeconómicas utilizadas en el modelo", fontsize=14)
axes[-1].set_xlabel("Fecha")

fig.tight_layout(rect=[0, 0, 1, 0.96])

png_path = output_dir / "macro_variables_2010_2024.png"
pdf_path = output_dir / "macro_variables_2010_2024.pdf"

fig.savefig(png_path, dpi=300, bbox_inches="tight")
fig.savefig(pdf_path, bbox_inches="tight")

print(f"Figura guardada en: {png_path}")
print(f"Figura guardada en: {pdf_path}")