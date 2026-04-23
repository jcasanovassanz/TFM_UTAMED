import numpy as np
import pandas as pd


def cumulative_return(returns: pd.Series) -> float:
    """
    Calcula la rentabilidad acumulada total de una serie de retornos.
    """
    returns = returns.dropna()
    if returns.empty:
        return np.nan
    return (1 + returns).prod() - 1


def annualized_return(returns: pd.Series, periods_per_year: int = 12) -> float:
    """
    Calcula la rentabilidad anualizada a partir de una serie de retornos periódicos.
    """
    returns = returns.dropna()
    if returns.empty:
        return np.nan

    cumulative = (1 + returns).prod()
    n_periods = len(returns)

    if n_periods == 0 or cumulative <= 0:
        return np.nan

    return cumulative ** (periods_per_year / n_periods) - 1


def annualized_volatility(returns: pd.Series, periods_per_year: int = 12) -> float:
    """
    Calcula la volatilidad anualizada.
    """
    returns = returns.dropna()
    if returns.empty:
        return np.nan

    return returns.std(ddof=1) * np.sqrt(periods_per_year)


def sharpe_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 12
) -> float:
    """
    Calcula el Sharpe ratio anualizado.
    risk_free_rate se interpreta como tasa anual.
    """
    ann_return = annualized_return(returns, periods_per_year=periods_per_year)
    ann_vol = annualized_volatility(returns, periods_per_year=periods_per_year)

    if pd.isna(ann_return) or pd.isna(ann_vol) or ann_vol == 0:
        return np.nan

    return (ann_return - risk_free_rate) / ann_vol


def max_drawdown(returns: pd.Series) -> float:
    """
    Calcula el máximo drawdown de la serie de retornos.
    """
    returns = returns.dropna()
    if returns.empty:
        return np.nan

    cumulative = (1 + returns).cumprod()
    rolling_max = cumulative.cummax()
    drawdown = cumulative / rolling_max - 1

    return drawdown.min()


def portfolio_metrics(
    returns: pd.Series,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 12
) -> dict:
    """
    Devuelve todas las métricas financieras principales en un diccionario.
    """
    return {
        "cumulative_return": cumulative_return(returns),
        "annualized_return": annualized_return(returns, periods_per_year=periods_per_year),
        "annualized_volatility": annualized_volatility(returns, periods_per_year=periods_per_year),
        "sharpe_ratio": sharpe_ratio(
            returns,
            risk_free_rate=risk_free_rate,
            periods_per_year=periods_per_year
        ),
        "max_drawdown": max_drawdown(returns),
    }