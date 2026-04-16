import pandas as pd

def load_macro():
    macro = pd.read_csv("macro.csv")

    macro = macro[macro["date"].notna()]
    macro = macro[macro["date"] != "..."]

    macro["date"] = pd.to_datetime(macro["date"], errors="coerce")
    macro = macro.dropna(subset=["date"])

    # Normalizar al primer día de cada mes
    macro["date"] = macro["date"].dt.to_period("M").dt.to_timestamp()

    macro = macro.sort_values("date")

    macro["inflation"] = macro["inflation"].shift(1)
    macro["interest_rate"] = macro["interest_rate"].shift(1)
    macro["unemployment"] = macro["unemployment"].shift(1)

    return macro