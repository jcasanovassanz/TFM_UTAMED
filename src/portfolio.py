import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier


def simulate_portfolio(df, model_func, features, top_pct=0.2, start_test_year=2015):
    """
    Simulate a monthly portfolio by selecting assets with highest predicted
    probability of outperforming the index.
    """
    df = df.copy()

    df = df.sort_values(["ticker", "date"])
    df["future_return"] = df.groupby("ticker")["return_1m"].shift(-1)
    df = df.dropna(subset=["future_return"])
    df["year"] = df["date"].dt.year

    years = sorted(df["year"].unique())
    portfolio_returns = []

    for year in years:
        if year < start_test_year:
            continue

        train = df[df["year"] < year]
        test = df[df["year"] == year].copy()

        if train.empty or test.empty:
            continue

        X_train = train[features]
        y_train = train["target"]
        X_test = test[features]

        model = model_func()
        model.fit(X_train, y_train)

        test["proba"] = model.predict_proba(X_test)[:, 1]
        test = test.sort_values(["date", "proba"], ascending=[True, False])
        test["rank"] = test.groupby("date")["proba"].rank(
            pct=True,
            ascending=False,
            method="first",
        )

        selected = test[test["rank"] <= top_pct]
        monthly_returns = selected.groupby("date")["future_return"].mean()
        portfolio_returns.append(monthly_returns)

    if not portfolio_returns:
        return pd.Series(dtype=float, name="portfolio_return")

    portfolio_returns = pd.concat(portfolio_returns).sort_index()
    portfolio_returns.name = "portfolio_return"
    return portfolio_returns


def get_logistic():
    return LogisticRegression(max_iter=1000, class_weight="balanced")


def get_rf():
    return RandomForestClassifier(
        n_estimators=200,
        max_depth=5,
        min_samples_leaf=1,
        random_state=42,
    )


def get_xgb():
    return XGBClassifier(
        n_estimators=50,
        max_depth=2,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric="logloss",
    )
