"""Herramientas para analizar tendencias en series temporales anuales.

Dependencias:
    pip install numpy scipy pymannkendall

Uso:
    resultado = trend_report(years, values)
    print(resultado.sen_slope_decade)
    print(resultado.trend)

Notas metodológicas:
- La pendiente de Sen y su intervalo se calculan usando los años reales como x,
  por lo que los huecos entre años sí afectan la pendiente por unidad de tiempo.
- pyMannKendall recibe únicamente la secuencia de valores. Si hay años faltantes,
  sus pruebas consideran observaciones consecutivas en el arreglo, no la distancia
  real entre años. La función agrega una advertencia cuando detecta esos huecos.
- La autocorrelación lag-1 se calcula sobre los residuos de la recta de Sen,
  no sobre la serie original, para reducir la influencia de la tendencia.
- |autocorrelación residual lag-1| > 0.2 es un criterio práctico configurable,
  no un umbral estadístico universal.
- Para advertir sobre un efecto pequeño se compara el cambio estimado con la
  desviación estándar de los residuos, no con la dispersión de la serie tendencial.
- _MIN_N_FOR_AUTOCORR = 4 es solo un mínimo computacional heurístico: produce tres
  parejas rezagadas, una estimación muy inestable que debe interpretarse con cautela.
  Los criterios se explican en README.md.
"""

from dataclasses import dataclass, field

import numpy as np
from scipy.stats import theilslopes
import pymannkendall as mk


# Criterios heurísticos del ejercicio (no son estándares universales).
_AUTOCORR_THRESHOLD = 0.2
_SMALL_EFFECT_RATIO = 0.1
# Con n=4 quedan solo 3 parejas (t-1, t); es un mínimo heurístico para poder
# calcular la correlación, NO una garantía de precisión estadística. Véase README.md.
_MIN_N_FOR_AUTOCORR = 4


@dataclass
class TrendResult:
    """Contiene el resumen estadístico de una tendencia."""

    n: int
    sen_slope_decade: float       # unidades por década
    ci_low_decade: float          # límite inferior, unidades por década
    ci_high_decade: float         # límite superior, unidades por década
    p_value: float
    z: float
    trend: str                    # "increasing" | "decreasing" | "no trend"
    significant: bool
    lag1_autocorr: float
    method: str                   # "Mann-Kendall" o "Hamed-Rao"
    warnings: list[str] = field(default_factory=list)


def _calculate_lag1_autocorrelation(values: np.ndarray) -> float:
    """Calcula corr(values[:-1], values[1:]); devuelve NaN si no se puede."""
    if values.size < _MIN_N_FOR_AUTOCORR:
        return float("nan")

    previous = values[:-1]
    following = values[1:]

    # La correlación de Pearson no está definida si uno de los arreglos es constante.
    if np.std(previous) == 0 or np.std(following) == 0:
        return float("nan")

    correlation = np.corrcoef(previous, following)[0, 1]
    return float(correlation) if np.isfinite(correlation) else float("nan")


def trend_report(years, values, alpha: float = 0.05) -> TrendResult:
    """Analiza una tendencia monotónica en una serie temporal anual.

    Parameters
    ----------
    years:
        Años de las observaciones. Pueden existir años faltantes.
    values:
        Valores observados para cada año. Se permiten NaN.
    alpha:
        Nivel de significancia de Mann-Kendall, entre 0 y 1 (por defecto 0.05).

    Returns
    -------
    TrendResult
        Pendiente de Sen e intervalo por década, resultados de la prueba,
        autocorrelación estimada, método empleado y advertencias.

    Raises
    ------
    ValueError
        Si alpha no está entre 0 y 1, las entradas no son vectores de igual
        longitud, hay años duplicados o quedan menos de dos observaciones válidas.
    """
    if not 0 < alpha < 1:
        raise ValueError("alpha debe ser un número entre 0 y 1 (sin incluirlos).")

    try:
        years_array = np.asarray(years, dtype=float)
        values_array = np.asarray(values, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("years y values deben contener datos numéricos.") from exc

    if years_array.ndim != 1 or values_array.ndim != 1:
        raise ValueError("years y values deben ser arreglos unidimensionales.")

    if years_array.size != values_array.size:
        raise ValueError("years y values deben tener la misma longitud.")

    # Quitar las parejas en las que el año o el valor no sea finito.
    valid_mask = np.isfinite(years_array) & np.isfinite(values_array)
    years_array = years_array[valid_mask]
    values_array = values_array[valid_mask]

    n = int(values_array.size)
    if n < 2:
        raise ValueError(
            "Se necesitan al menos dos observaciones válidas para calcular la pendiente."
        )

    # Ordenar ambas series usando el mismo índice para conservar las parejas año-valor.
    order = np.argsort(years_array)
    years_array = years_array[order]
    values_array = values_array[order]

    if np.any(np.diff(years_array) == 0):
        raise ValueError("years contiene años duplicados; debe haber un valor por año.")

    warning_messages: list[str] = []

    if n < 10:
        warning_messages.append(
            "Serie corta: hay menos de 10 observaciones válidas; interpreta el resultado con cautela."
        )

    # Avisar si hay huecos: las pruebas Mann-Kendall reciben la secuencia de valores,
    # mientras que theilslopes sí utiliza las distancias reales entre años.
    if np.any(np.diff(years_array) != 1):
        warning_messages.append(
            "Hay huecos en years: la autocorrelación y Mann-Kendall usan observaciones "
            "válidas consecutivas, que pueden no corresponder a años calendario consecutivos."
        )

    # SciPy interpreta alpha como nivel de confianza (p. ej. 0.95), mientras que
    # esta función usa alpha como nivel de significancia (p. ej. 0.05).
    confidence_level = 1.0 - alpha
    slope, intercept, ci_low, ci_high = theilslopes(
        values_array,
        years_array,
        alpha=confidence_level,
    )

    # theilslopes entrega unidades por año; multiplicar por 10 da unidades por década.
    sen_slope_decade = float(slope * 10.0)
    ci_low_decade = float(ci_low * 10.0)
    ci_high_decade = float(ci_high * 10.0)

    # Quitar la tendencia estimada antes de medir autocorrelación. Si se calcula
    # sobre values_array directamente, una tendencia sostenida puede producir una
    # correlación alta aunque el ruido/residuo no tenga memoria temporal.
    fitted_trend = slope * years_array + intercept
    residuals = values_array - fitted_trend
    lag1_autocorr = _calculate_lag1_autocorrelation(residuals)
    if not np.isfinite(lag1_autocorr):
        warning_messages.append(
            "No fue posible estimar la autocorrelación lag-1 residual de forma fiable "
            "(se requieren al menos 4 observaciones y variación en los residuos); "
            "se utilizará Mann-Kendall original."
        )

    # Seleccionar la prueba según la regla práctica indicada en el ejercicio.
    if (
        np.isfinite(lag1_autocorr)
        and abs(lag1_autocorr) > _AUTOCORR_THRESHOLD
    ):
        test_result = mk.hamed_rao_modification_test(values_array, alpha=alpha)
        method = "Hamed-Rao"
    else:
        test_result = mk.original_test(values_array, alpha=alpha)
        method = "Mann-Kendall"

    p_value = float(test_result.p)
    z = float(test_result.z)
    trend = str(test_result.trend)
    significant = bool(p_value < alpha)

    # Advertir cuando hay significancia estadística, pero el cambio estimado
    # durante el periodo completo es pequeño respecto a la variabilidad observada.
    # La variabilidad relevante para esta advertencia es la dispersión alrededor
    # de la tendencia, por eso usamos los residuos y no la serie original.
    variability = float(np.std(residuals, ddof=1))
    elapsed_years = float(years_array[-1] - years_array[0])
    estimated_total_change = abs(sen_slope_decade) * elapsed_years / 10.0

    if (
        significant
        and np.isfinite(variability)
        and variability > 0
        and estimated_total_change < _SMALL_EFFECT_RATIO * variability
    ):
        warning_messages.append(
            "La tendencia es estadísticamente significativa, pero el cambio estimado "
            "durante todo el periodo es menor al 10 % de la desviación estándar; "
            "su relevancia práctica podría ser pequeña."
        )

    return TrendResult(
        n=n,
        sen_slope_decade=sen_slope_decade,
        ci_low_decade=ci_low_decade,
        ci_high_decade=ci_high_decade,
        p_value=p_value,
        z=z,
        trend=trend,
        significant=significant,
        lag1_autocorr=float(lag1_autocorr),
        method=method,
        warnings=warning_messages,
    )