import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier


def simulate_portfolio(df, model_func, features, top_pct=0.2, start_test_year=2015):
    """
    Simula una cartera mensual seleccionando las acciones con mayor probabilidad
    estimada de batir al índice.

    La rentabilidad de la cartera se calcula con la rentabilidad futura t+1,
    no con la rentabilidad contemporánea, para evitar incoherencias temporales.
    """

    df = df.copy()

    # Asegurar orden temporal correcto
    df = df.sort_values(["ticker", "date"])

    # Crear rentabilidad futura real de cada acción
    df["future_return"] = df.groupby("ticker")["return_1m"].shift(-1)

    # Eliminar observaciones sin retorno futuro
    df = df.dropna(subset=["future_return"])

    # Crear año para walk-forward
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

        # Entrenar modelo solo con datos pasados
        model = model_func()
        model.fit(X_train, y_train)

        # Probabilidad de clase positiva: acción bate al índice
        test["proba"] = model.predict_proba(X_test)[:, 1]

        # Ordenar por fecha y probabilidad descendente
        test = test.sort_values(["date", "proba"], ascending=[True, False])

        # Ranking porcentual por mes
        test["rank"] = test.groupby("date")["proba"].rank(
            pct=True,
            ascending=False,
            method="first"
        )

        # Seleccionar top %
        selected = test[test["rank"] <= top_pct]

        # Rentabilidad media futura de las acciones seleccionadas
        monthly_returns = selected.groupby("date")["future_return"].mean()

        portfolio_returns.append(monthly_returns)

    if not portfolio_returns:
        return pd.Series(dtype=float, name="portfolio_return")

    portfolio_returns = pd.concat(portfolio_returns).sort_index()
    portfolio_returns.name = "portfolio_return"

    return portfolio_returns


def calculate_portfolio_metrics(portfolio_returns):
    """
    Calcula métricas financieras básicas de la cartera.
    Frecuencia mensual. Sharpe calculado con tipo libre de riesgo igual a cero.
    """

    portfolio_returns = portfolio_returns.dropna()

    if portfolio_returns.empty:
        return {
            "cumulative_return": None,
            "annualized_volatility": None,
            "sharpe_ratio": None,
            "max_drawdown": None,
        }

    cumulative_curve = (1 + portfolio_returns).cumprod()

    cumulative_return = cumulative_curve.iloc[-1] - 1

    annualized_volatility = portfolio_returns.std() * (12 ** 0.5)

    if portfolio_returns.std() != 0:
        sharpe_ratio = (portfolio_returns.mean() / portfolio_returns.std()) * (12 ** 0.5)
    else:
        sharpe_ratio = None

    running_max = cumulative_curve.cummax()
    drawdown = cumulative_curve / running_max - 1
    max_drawdown = drawdown.min()

    return {
        "cumulative_return": cumulative_return,
        "annualized_volatility": annualized_volatility,
        "sharpe_ratio": sharpe_ratio,
        "max_drawdown": max_drawdown,
    }


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