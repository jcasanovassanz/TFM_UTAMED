import pandas as pd

def load_macro():
    macro = pd.read_csv("data/macro/macro.csv")
    macro["date"] = pd.to_datetime(macro["date"])
    return macro