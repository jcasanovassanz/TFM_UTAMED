import pandas as pd


def simulate_portfolio(df, model_func, features, top_pct=0.2, start_test_year=2015):

    df = df.copy()
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
        y_test = test["target"]

        # entrenar modelo
        model = model_func()
        model.fit(X_train, y_train)

        # añadir probabilidades al dataframe completo de test
        test["proba"] = model.predict_proba(X_test)[:, 1]

        # ordenar por fecha y probabilidad
        test = test.sort_values(["date", "proba"], ascending=[True, False])

        # ranking por fecha
        test["rank"] = test.groupby("date")["proba"].rank(
            pct=True,
            ascending=False,
            method="first"
        )

        # seleccionar top %
        selected = test[test["rank"] <= top_pct]

        # rentabilidad media mensual de la cartera
        monthly_returns = selected.groupby("date")["return_1m"].mean()

        portfolio_returns.append(monthly_returns)

    portfolio_returns = pd.concat(portfolio_returns).sort_index()
    portfolio_returns.name = "portfolio_return"

    return portfolio_returns

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier


def get_logistic():
    return LogisticRegression(max_iter=1000, class_weight="balanced")


def get_rf():
    return RandomForestClassifier(
        n_estimators=200,
        max_depth=5,
        min_samples_leaf=1,
        random_state=42
    )


def get_xgb():
    return XGBClassifier(
        n_estimators=50,
        max_depth=2,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric="logloss"
    )