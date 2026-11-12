# Trend Detective

Herramienta para detectar tendencias en variables del sistema terrestre.

**NASA Space Apps Challenge 2026: Be An Earth System Trend Detective!**

## Cómo correrlo

### 1. Crear entorno virtual

```powershell
python -m venv .venv
```

### 2. Activar entorno virtual

```powershell
.venv\Scripts\Activate.ps1
```

### 3. Instalar dependencias

```powershell
pip install -r requirements.txt
```

### 4. Ejecutar aplicación

```powershell
streamlit run app/main.py
```

---

## Análisis estadístico de tendencias

El módulo `stats/trends.py` contiene la función `trend_report`, diseñada para analizar tendencias en series temporales anuales.

### Métodos utilizados

- **Pendiente de Sen:** estima la magnitud y dirección de la tendencia mediante la mediana de las pendientes entre pares de observaciones.
- **Intervalo de confianza de Sen:** estima el rango de valores plausibles para la pendiente mediante `scipy.stats.theilslopes`.
- **Mann-Kendall:** evalúa si existe una tendencia monotónica en la serie temporal.
- **Hamed-Rao:** aplica una modificación de Mann-Kendall para considerar la autocorrelación serial.

### Uso

```python
from stats.trends import trend_report

years = [2000, 2001, 2002, 2003, 2004, 2005]
values = [10.2, 10.5, 10.8, 11.1, 11.4, 11.7]

result = trend_report(years, values)

print("Tendencia:", result.trend)
print("Pendiente por década:", result.sen_slope_decade)
print("Valor p:", result.p_value)
print("Significativa:", result.significant)
```

La función recibe:

| Parámetro | Descripción |
|---|---|
| `years` | Años correspondientes a las observaciones. |
| `values` | Valores de la variable analizada. |
| `alpha` | Nivel de significancia estadística (por defecto, 0.05). |

El resultado `TrendResult` contiene:

- `n`: número de observaciones válidas.
- `sen_slope_decade`: pendiente de Sen por década.
- `ci_low_decade` y `ci_high_decade`: límites del intervalo de confianza.
- `p_value`: valor p de la prueba seleccionada.
- `z`: estadístico Z.
- `trend`: dirección de la tendencia.
- `significant`: indica si el resultado es estadísticamente significativo.
- `lag1_autocorr`: autocorrelación residual de retardo 1.
- `method`: método estadístico utilizado.
- `warnings`: advertencias del análisis.

### Consideraciones metodológicas

1. Los valores no finitos (`NaN` e infinitos) se eliminan conjuntamente con sus años correspondientes.
2. La pendiente y el intervalo de confianza se expresan en unidades por década.
3. La autocorrelación lag-1 se calcula sobre los residuos de la tendencia estimada, evitando confundir una tendencia ascendente o descendente con autocorrelación residual.
4. Se utiliza un umbral heurístico de `|r1| > 0.2` para decidir cuándo aplicar Hamed-Rao. No representa una prueba formal de significancia de autocorrelación ni es un umbral universal.
5. Se requiere un mínimo computacional de cuatro observaciones para estimar la autocorrelación, aunque una muestra tan pequeña no garantiza una estimación confiable.
6. Las observaciones deben representar una secuencia temporal adecuada. Los años faltantes pueden afectar la interpretación de la autocorrelación y de las pruebas que consideran observaciones consecutivas.

La significancia estadística no implica necesariamente que la magnitud de la tendencia sea importante desde el punto de vista ambiental.

---

## Pruebas automatizadas

Las pruebas del módulo se encuentran en `stats/tests/test_trends.py`.

Para ejecutarlas desde la raíz del proyecto:

```powershell
python -m pytest -v
```

Los casos de prueba incluyen:

- Tendencia creciente conocida.
- Ruido blanco.
- Serie AR(1) sin tendencia determinista.
- Validación de `alpha`.
- Detección de años duplicados.
- Eliminación de valores `NaN`.

---

## Fuentes de datos

Fetterer, F., Knowles, K., Meier, W. N., Savoie, M., Windnagel, A. K., & Stafford, T. (2025). Sea Ice Index (G02135, Version 4) [Conjunto de datos]. National Snow and Ice Data Center. https://doi.org/10.7265/a98x-0f50

GISTEMP Team. (2026). GISS Surface Temperature Analysis (GISTEMP), version 4 [Conjunto de datos]. NASA Goddard Institute for Space Studies. Consultado el 11 de noviembre de 2026. https://data.giss.nasa.gov/gistemp/

Lenssen, N., Schmidt, G. A., Hendrickson, M., Jacobs, P., Menne, M. J., & Ruedy, R. (2024). A NASA GISTEMPv4 observational uncertainty ensemble. Journal of Geophysical Research: Atmospheres, 129, e2023JD040179. https://doi.org/10.1029/2023JD040179
