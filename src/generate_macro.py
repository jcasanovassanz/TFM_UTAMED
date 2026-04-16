import pandas as pd
import numpy as np

# -----------------------
# CONFIG
# -----------------------

start = "2010-01-01"
end = "2024-11-01"

# -----------------------
# GENERAR FECHAS
# -----------------------

dates = pd.date_range(start=start, end=end, freq="MS")

# -----------------------
# GENERAR VARIABLES SIMULADAS
# -----------------------

np.random.seed(42)

inflation = 2 + np.sin(np.linspace(0, 20, len(dates))) + np.random.normal(0, 0.2, len(dates))
interest_rate = 1 + np.clip(np.cumsum(np.random.normal(0, 0.05, len(dates))), -1, 3)
unemployment = 8 + np.cos(np.linspace(0, 10, len(dates))) + np.random.normal(0, 0.3, len(dates))

# -----------------------
# DATAFRAME
# -----------------------

macro = pd.DataFrame({
    "date": dates,
    "inflation": inflation,
    "interest_rate": interest_rate,
    "unemployment": unemployment
})

# Redondear
macro["inflation"] = macro["inflation"].round(2)
macro["interest_rate"] = macro["interest_rate"].round(2)
macro["unemployment"] = macro["unemployment"].round(2)

# -----------------------
# EXPORTAR CSV
# -----------------------

macro.to_csv("macro.csv", index=False)

print("macro.csv generado correctamente")
print(macro.head())