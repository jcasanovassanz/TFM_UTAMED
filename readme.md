Aquí tienes el contenido propuesto del `README.md`, sin escribir ningún archivo:

```markdown
# TFM_UTAMED

Repositorio técnico para la construcción de un dataset panel financiero-macroeconómico, entrenamiento de modelos de clasificación y simulación de una estrategia de cartera basada en predicciones de exceso de rentabilidad.

El proyecto trabaja con activos europeos, datos mensuales de mercado descargados mediante `yfinance`, variables macroeconómicas reales y validación walk-forward para comparar modelos predictivos.

## 1. Descripción general del proyecto

El objetivo del repositorio es evaluar si variables de mercado y macroeconómicas ayudan a predecir si un activo tendrá una rentabilidad futura superior al índice de referencia.

El flujo principal es:

1. Preparar variables macroeconómicas reales.
2. Descargar precios mensuales de activos e índice mediante `yfinance`.
3. Construir un dataset panel por fecha y ticker.
4. Generar features de momentum, volatilidad, exceso de retorno y ranking cross-sectional.
5. Entrenar modelos con validación walk-forward.
6. Comparar resultados predictivos.
7. Simular una cartera seleccionando los activos con mayor probabilidad estimada.

## 2. Estructura del repositorio

```text
TFM_UTAMED/
├── data/
│   ├── raw/
│   │   ├── inflation.csv
│   │   ├── interest_rate.csv
│   │   └── unemployment.csv
│   ├── processed/
│   │   ├── macro_real.csv
│   │   └── panel_dataset_v1.csv
│   └── macro/
│       └── macro.csv
├── notebooks/
│   ├── 01_exploracion.ipynb
│   ├── 02_features.ipynb
│   ├── 03_baseline_logistic.ipynb
│   ├── 04_random_forest.ipynb
│   └── 05_xgboost.ipynb
├── results/
│   ├── logistic_market_by_year.csv
│   ├── logistic_macro_by_year.csv
│   ├── rf_market_by_year.csv
│   ├── rf_macro_by_year.csv
│   ├── xgb_market_by_year.csv
│   ├── xgb_macro_by_year.csv
│   └── model_comparison_summary.csv
└── src/
    ├── build_macro_real.py
    ├── build_dataset.py
    ├── feature_engineering.py
    ├── macro_loader.py
    ├── model.py
    ├── run_baseline.py
    ├── tune_models.py
    ├── portfolio.py
    ├── metrics.py
    ├── run_portfolio.py
    ├── data_loader.py
    └── generate_macro.py
```

## 3. Flujo de ejecución recomendado

El pipeline recomendado es el siguiente:

```bash
python src/build_macro_real.py
python src/build_dataset.py
python src/tune_models.py
python src/run_baseline.py
python src/run_portfolio.py
```

Orden lógico:

1. `build_macro_real.py`: transforma los CSV macroeconómicos reales.
2. `build_dataset.py`: descarga precios, calcula retornos, genera target y crea el panel final.
3. `tune_models.py`: prueba combinaciones de hiperparámetros para Random Forest y XGBoost.
4. `run_baseline.py`: entrena y compara modelos finales.
5. `run_portfolio.py`: simula una cartera basada en las señales de los modelos.

## 4. Archivos principales

### `src/build_macro_real.py`

Carga los datos macroeconómicos desde `data/raw/`, limpia nombres de columnas, normaliza fechas mensuales y genera:

```text
data/processed/macro_real.csv
```

Variables macro utilizadas:

- `inflation`
- `interest_rate`
- `unemployment`

### `src/build_dataset.py`

Script principal de construcción del dataset panel.

Funciones principales:

- Descarga precios mensuales de acciones europeas mediante `yfinance`.
- Descarga el índice `^STOXX50E`.
- Calcula retornos mensuales.
- Calcula retorno relativo frente al índice.
- Define el target binario:
  - `1`: el activo supera al índice en el mes siguiente.
  - `0`: el activo no supera al índice.
- Añade features mediante `feature_engineering.py`.
- Incorpora variables macroeconómicas reales.
- Exporta:

```text
data/processed/panel_dataset_v1.csv
```

### `src/feature_engineering.py`

Genera variables predictivas basadas en información histórica:

- `return_lag1`
- `ret_3m`
- `ret_6m`
- `vol_3m`
- `vol_6m`
- `excess_return`
- `rank_ret_3m`

### `src/macro_loader.py`

Carga `data/processed/macro_real.csv` y convierte la columna `date` a formato fecha.

### `src/model.py`

Contiene las funciones de validación walk-forward para:

- Regresión logística.
- Random Forest.
- XGBoost.

Métricas calculadas por año:

- `accuracy`
- `precision`
- `recall`
- `f1`
- `auc`

### `src/tune_models.py`

Ejecuta una búsqueda manual de hiperparámetros para Random Forest y XGBoost usando el dataset panel.

### `src/run_baseline.py`

Ejecuta la comparativa principal de modelos:

- Logistic Regression con features de mercado.
- Logistic Regression con mercado + macro.
- Random Forest con features de mercado.
- Random Forest con mercado + macro.
- XGBoost con features de mercado.
- XGBoost con mercado + macro.

Exporta los resultados a `results/`.

### `src/portfolio.py`

Simula una cartera seleccionando, en cada fecha, el porcentaje superior de activos según la probabilidad predicha por el modelo.

Por defecto, selecciona el top 20%.

### `src/metrics.py`

Calcula métricas financieras de la cartera:

- Rentabilidad acumulada.
- Rentabilidad anualizada.
- Volatilidad anualizada.
- Sharpe ratio.
- Máximo drawdown.

### `src/run_portfolio.py`

Ejecuta la simulación de cartera para:

- Logistic Regression.
- Random Forest.
- XGBoost.

Actualmente imprime las métricas por consola.

## 5. Datos de entrada y salida

### Datos de entrada

Los datos macroeconómicos reales deben estar en:

```text
data/raw/inflation.csv
data/raw/interest_rate.csv
data/raw/unemployment.csv
```

Los datos de mercado se descargan automáticamente mediante `yfinance`.

Activos incluidos en el dataset:

```text
SAP.DE, ASML.AS, MC.PA, OR.PA, AIR.PA, SAN.PA, BNP.PA, AI.PA,
SU.PA, KER.PA, BAS.DE, ALV.DE, ADS.DE, IFX.DE, DTE.DE,
ENEL.MI, ENI.MI, ISP.MI, IBE.MC, SAN.MC
```

Índice de referencia:

```text
^STOXX50E
```

Periodo configurado:

```text
2010-01-01 a 2024-12-31
```

### Datos de salida

Dataset macro procesado:

```text
data/processed/macro_real.csv
```

Dataset panel final:

```text
data/processed/panel_dataset_v1.csv
```

Resultados de modelos:

```text
results/model_comparison_summary.csv
results/logistic_market_by_year.csv
results/logistic_macro_by_year.csv
results/rf_market_by_year.csv
results/rf_macro_by_year.csv
results/xgb_market_by_year.csv
results/xgb_macro_by_year.csv
```

## 6. Comandos para reproducir el experimento

Crear entorno virtual:

```bash
python -m venv .venv
```

Activar entorno en Windows:

```bash
.venv\Scripts\activate
```

Instalar dependencias principales:

```bash
pip install pandas numpy scikit-learn xgboost yfinance
```

Ejecutar pipeline completo:

```bash
python src/build_macro_real.py
python src/build_dataset.py
python src/tune_models.py
python src/run_baseline.py
python src/run_portfolio.py
```

## 7. Advertencias técnicas

- `yfinance` depende de conexión a internet y de la disponibilidad de la API/servicio de Yahoo Finance. Los resultados pueden variar si hay cambios en tickers, columnas devueltas o disponibilidad histórica.
- `src/data_loader.py` está obsoleto. Contiene una versión antigua del flujo y referencias a módulos antiguos como `features` y `macro`. No pertenece al pipeline final recomendado.
- `src/generate_macro.py` generaba una macro sintética y no forma parte del pipeline final. El flujo actual usa datos macro reales desde `data/raw/` y genera `data/processed/macro_real.csv`.
- Los notebooks en `notebooks/` están vacíos actualmente y no deben considerarse documentación ejecutable del experimento.

## 8. Resultados principales esperados

Al reproducir el pipeline deberían obtenerse tres bloques principales de resultados:

1. Dataset panel

   Archivo esperado:

   ```text
   data/processed/panel_dataset_v1.csv
   ```

   Contiene observaciones mensuales por `date` y `ticker`, retornos, índice de referencia, features de mercado, variables macro y target binario.

2. Comparativa de modelos

   Archivo esperado:

   ```text
   results/model_comparison_summary.csv
   ```

   Resume el rendimiento medio de Logistic Regression, Random Forest y XGBoost con dos conjuntos de variables:

   - Solo mercado.
   - Mercado + macro.

3. Simulación de cartera

   Ejecutada mediante:

   ```bash
   python src/run_portfolio.py
   ```

   Calcula métricas financieras para carteras basadas en las predicciones de cada modelo.

## 9. Próximos pasos técnicos

- Añadir un benchmark financiero explícito, por ejemplo una cartera equal-weight o el índice `^STOXX50E`.
- Exportar las métricas financieras de `run_portfolio.py` a CSV en `results/`.
- Generar gráficos de equity curve para comparar visualmente la evolución de las carteras simuladas frente al benchmark.
```