import pandas as pd

# -----------------------
# CARGAR ARCHIVOS
# -----------------------

inflation = pd.read_csv("data/raw/inflation.csv")
unemployment = pd.read_csv("data/raw/unemployment.csv")
rates = pd.read_csv("data/raw/interest_rate.csv")

# -----------------------
# LIMPIAR INFLATION
# -----------------------

inflation = inflation.rename(columns={
    "DATE": "date",
    inflation.columns[-1]: "inflation"
})

inflation = inflation[["date", "inflation"]]
inflation["date"] = pd.to_datetime(inflation["date"])
inflation["date"] = inflation["date"].dt.to_period("M").dt.to_timestamp()

# -----------------------
# LIMPIAR UNEMPLOYMENT
# -----------------------

unemployment = unemployment.rename(columns={
    unemployment.columns[0]: "date",
    unemployment.columns[-1]: "unemployment"
})

unemployment = unemployment[["date", "unemployment"]]
unemployment["date"] = pd.to_datetime(unemployment["date"])
unemployment["date"] = unemployment["date"].dt.to_period("M").dt.to_timestamp()

# -----------------------
# LIMPIAR INTEREST RATE
# -----------------------

rates = rates.rename(columns={
    rates.columns[0]: "date",
    rates.columns[-1]: "interest_rate"
})

rates = rates[["date", "interest_rate"]]
rates["date"] = pd.to_datetime(rates["date"])
rates["date"] = rates["date"].dt.to_period("M").dt.to_timestamp()

# -----------------------
# MERGE
# -----------------------

macro = inflation.merge(unemployment, on="date", how="inner")
macro = macro.merge(rates, on="date", how="inner")

# -----------------------
# FILTRO FECHAS
# -----------------------

macro = macro[macro["date"] >= "2010-01-01"]
macro = macro[macro["date"] <= "2026-04-01"]

# -----------------------
# ORDENAR Y LIMPIAR
# -----------------------

macro = macro.sort_values("date")
macro = macro.dropna()

# -----------------------
# CHECK
# -----------------------

print("\nMacro final:")
print(macro.head())
print(macro.info())

# -----------------------
# EXPORTAR
# -----------------------

macro.to_csv("data/processed/macro_real.csv", index=False)

print("\nmacro_real.csv generado correctamente")

macro = macro.reset_index(drop=True)
print(macro.describe())