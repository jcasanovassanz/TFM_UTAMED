# TFM_UTAMED

Repositorio tecnico para construccion de dataset panel financiero-macroeconomico, entrenamiento de modelos de clasificacion y simulacion de carteras basadas en predicciones.

## 1. Objetivo

Evaluar si variables de mercado y macroeconomicas ayudan a predecir si un activo batira al indice en el siguiente periodo, y usar esas probabilidades para construir carteras long-only con rebalanceo mensual.

## 2. Estructura principal

```text
TFM_UTAMED/
|-- data/
|   |-- raw/
|   |-- processed/
|       |-- macro_real.csv
|       `-- panel_dataset_v1.csv
|-- results/
|   |-- model_comparison_summary.csv
|   |-- logistic_market_by_year.csv
|   |-- logistic_macro_by_year.csv
|   |-- rf_market_by_year.csv
|   |-- rf_macro_by_year.csv
|   |-- xgb_market_by_year.csv
|   |-- xgb_macro_by_year.csv
|   `-- portfolio/
|       |-- portfolio_metrics.csv
|       |-- portfolio_returns.csv
|       `-- cumulative_returns.csv
`-- src/
    |-- build_macro_real.py
    |-- build_dataset.py
    |-- feature_engineering.py
    |-- macro_loader.py
    |-- model.py
    |-- tune_models.py
    |-- run_baseline.py
    |-- portfolio.py
    |-- metrics.py
    `-- run_portfolio.py
```

## 3. Flujo recomendado

```bash
python src/build_macro_real.py
python src/build_dataset.py
python src/tune_models.py
python src/run_baseline.py
python src/run_portfolio.py
```

## 4. Modelos predictivos (run_baseline.py)

`src/run_baseline.py`:

- Usa `data/processed/panel_dataset_v1.csv`.
- Aplica validacion walk-forward anual (`start_test_year=2015`) mediante `src/model.py`.
- Ejecuta 6 configuraciones:
  - Logistic Regression Market
  - Logistic Regression Market + Macro
  - Random Forest Market
  - Random Forest Market + Macro
  - XGBoost Market
  - XGBoost Market + Macro
- Guarda:
  - `results/model_comparison_summary.csv`
  - `results/*_by_year.csv` (6 archivos)

Features:

- Market:
  - `return_lag1`
  - `ret_3m`
  - `ret_6m`
  - `vol_3m`
  - `vol_6m`
  - `excess_return`
  - `rank_ret_3m`
- Market + Macro:
  - Market +
  - `inflation`
  - `interest_rate`
  - `unemployment`

## 5. Simulacion de cartera (run_portfolio.py)

`src/run_portfolio.py`:

- Usa `data/processed/panel_dataset_v1.csv`.
- Ejecuta las mismas 6 configuraciones (Market y Market + Macro para los 3 modelos).
- Usa `simulate_portfolio` de `src/portfolio.py` con:
  - `top_pct=0.2`
  - validacion walk-forward anual (`start_test_year=2015`)
  - seleccion long-only
  - equal weight (media simple de retornos seleccionados)
  - rebalanceo mensual
- Usa una unica implementacion oficial de metricas: `src/metrics.py`.
- Crea `results/portfolio` si no existe y sobrescribe:
  - `results/portfolio/portfolio_metrics.csv`
  - `results/portfolio/portfolio_returns.csv`
  - `results/portfolio/cumulative_returns.csv`
- Imprime una tabla resumen final en consola.

## 6. Metricas financieras

Definidas en `src/metrics.py`:

- `cumulative_return`
- `annualized_return`
- `annualized_volatility` (factor `sqrt(12)`)
- `sharpe_ratio` (con `risk_free_rate=0.0` por defecto)
- `max_drawdown`

## 7. Notas de mantenimiento

- `src/portfolio.py` contiene solo logica de simulacion y constructores de modelos.
- La logica de metricas de cartera esta centralizada en `src/metrics.py`.
- `src/portfolio_old.py` no forma parte del flujo oficial.

## 8. Reproducibilidad en Windows

Si `python` no esta disponible en `PATH`, ejecuta el pipeline con el interprete del entorno virtual:

```powershell
.venv\Scripts\python.exe src\build_macro_real.py
.venv\Scripts\python.exe src\build_dataset.py
.venv\Scripts\python.exe src\tune_models.py
.venv\Scripts\python.exe src\run_baseline.py
.venv\Scripts\python.exe src\run_portfolio.py
```

Comprobaciones rapidas:

```powershell
Import-Csv results\portfolio\portfolio_metrics.csv | Measure-Object
Get-ChildItem results\portfolio
```
