import yfinance as yf
import pandas as pd

# -----------------------
# CONFIG
# -----------------------

tickers = ["SAP.DE", "MC.PA"]  # empieza con pocos
start = "2010-01-01"
end = "2024-12-31"

# -----------------------
# DESCARGA DATOS
# -----------------------

raw_data = yf.download(
    tickers,
    start=start,
    end=end,
    interval="1mo",
    auto_adjust=False,
    progress=False,
    threads=False
)

print("Columnas acciones:")
print(raw_data.columns)

# Selección robusta de precios
price_level = raw_data.columns.get_level_values(0)

if "Adj Close" in price_level:
    data = raw_data["Adj Close"]
elif "Close" in price_level:
    data = raw_data["Close"]
else:
    raise ValueError("No se encontraron columnas de precios.")

# Eliminar tickers fallidos
data = data.dropna(axis=1, how="all")

print("\nPrecios acciones:")
print(data.head())

# -----------------------
# ÍNDICE
# -----------------------

raw_index = yf.download(
    "^STOXX50E",
    start=start,
    end=end,
    interval="1mo",
    auto_adjust=False,
    progress=False,
    threads=False
)

print("\nColumnas índice:")
print(raw_index.columns)

# Selección robusta índice
if isinstance(raw_index.columns, pd.MultiIndex):
    price_level_index = raw_index.columns.get_level_values(0)

    if "Adj Close" in price_level_index:
        index = raw_index["Adj Close"]
    elif "Close" in price_level_index:
        index = raw_index["Close"]
    else:
        raise ValueError("No se encontraron columnas de precios en índice.")
else:
    if "Adj Close" in raw_index.columns:
        index = raw_index["Adj Close"]
    elif "Close" in raw_index.columns:
        index = raw_index["Close"]
    else:
        raise ValueError("No se encontraron columnas de precios en índice.")

index = index.squeeze()

# -----------------------
# RETURNS
# -----------------------

returns = data.pct_change()
index_returns = index.pct_change()

# -----------------------
# PANEL DATASET
# -----------------------

df = returns.stack().reset_index()
df.columns = ["date", "ticker", "return_1m"]

# -----------------------
# MERGE INDEX
# -----------------------

index_df = index_returns.reset_index()
index_df.columns = ["date", "index_return"]

df = df.merge(index_df, on="date", how="left")

# -----------------------
# TARGET (t+1)
# -----------------------

df = df.sort_values(["ticker", "date"])

df["future_return"] = df.groupby("ticker")["return_1m"].shift(-1)
df["future_index_return"] = df.groupby("ticker")["index_return"].shift(-1)

df["future_excess_return"] = df["future_return"] - df["future_index_return"]

df["target"] = (df["future_excess_return"] > 0).astype(int)

# -----------------------
# LIMPIEZA
# -----------------------

df = df.dropna()

df = df.drop(columns=[
    "future_return",
    "future_index_return",
    "future_excess_return"
])

# -----------------------
# CHECK FINAL
# -----------------------

print("\nDataset final:")
print(df.head())

print("\nInfo:")
print(df.info())

print("\nDistribución target:")
print(df["target"].value_counts(normalize=True))

from features import add_features

df = add_features(df)

df = df.dropna()

# -----------------------
# CHECK GENERAL
# -----------------------

print(df.columns)
print(df.head())
print(df.isnull().sum())

# -----------------------
# VALIDACIÓN FEATURES
# -----------------------

print("\nCheck lag y momentum:")
print(df.groupby("ticker")[["return_1m", "return_lag1", "ret_3m", "ret_6m"]].head(5))

print("\nCheck volatilidad:")
print(df.groupby("ticker")[["vol_3m", "vol_6m"]].head(5))

print("\nCheck ranking:")
print(df.groupby("date")["rank_ret_3m"].describe().head())

# Cargar variables macro

from macro import load_macro

macro = load_macro()

print("\nMacro cargado:")
print(macro.head(10))
print(macro.info())

print("\nRango fechas macro:")
print(macro["date"].min(), macro["date"].max())

print("\nFechas ejemplo macro:")
print(macro["date"].head(10))

print("\nRango fechas df:")
print(df["date"].min(), df["date"].max())

print("\nFechas ejemplo df:")
print(df["date"].head(10))

# MERGE REAL
df = df.merge(macro, on="date", how="left")

print("\nColumnas tras merge:")
print(df.columns)

print("\nTras merge, antes de dropna:")
print(df[["date", "inflation", "interest_rate", "unemployment"]].head(15))

print("\nNaNs macro tras merge:")
print(df[["inflation", "interest_rate", "unemployment"]].isnull().sum())

print("\nNúmero de filas tras merge:")
print(len(df))





# Validacion de variables

print(df[["date", "inflation", "interest_rate", "unemployment"]].head())
print(df.isnull().sum())


features_market = [
    "return_lag1",
    "ret_3m",
    "ret_6m",
    "vol_3m",
    "vol_6m",
    "excess_return",
    "rank_ret_3m"
]

features_macro = features_market + [
    "inflation",
    "interest_rate",
    "unemployment"
]

from model import run_walk_forward_validation

results_market = run_walk_forward_validation(df, features_market, start_test_year=2015)
results_macro = run_walk_forward_validation(df, features_macro, start_test_year=2015)

print("\nResultados modelo solo mercado:")
print(results_market)

print("\nResultados modelo mercado + macro:")
print(results_macro)

print("\nMedia modelo solo mercado:")
print(results_market.mean(numeric_only=True))

print("\nMedia modelo mercado + macro:")
print(results_macro.mean(numeric_only=True))