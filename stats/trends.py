"""Tools for analyzing trends in annual time series.

Dependencies:
    pip install numpy scipy pymannkendall

Usage:
    resultado = trend_report(years, values)
    print(resultado.sen_slope_decade)
    print(resultado.trend)

Methodological notes:
- Sen's slope and its confidence interval are calculated using the actual years as x,
  so gaps between years affect the slope per unit of time.
- pyMannKendall receives only the sequence of values. If years are missing,
  its tests treat consecutive observations in the array as consecutive, regardless
  of the actual distance between years. The function adds a warning when it detects
  such gaps.
- Lag-1 autocorrelation is calculated on the residuals from the Sen line,
  not on the original series, to reduce the influence of the trend.
- |residual lag-1 autocorrelation| > 0.2 is a configurable practical criterion,
  not a universal statistical threshold.
- To warn about a small effect, the estimated change is compared with the
  standard deviation of the residuals, not the dispersion of the trended series.
- _MIN_N_FOR_AUTOCORR = 4 is only a heuristic computational minimum: it yields three
  lagged pairs, a highly unstable estimate that should be interpreted with caution.
  The criteria are explained in README.md.
"""

from dataclasses import dataclass, field

import numpy as np
from scipy.stats import theilslopes
import pymannkendall as mk


# Heuristic criteria for the exercise (not universal standards).
_AUTOCORR_THRESHOLD = 0.2
_SMALL_EFFECT_RATIO = 0.1
# With n=4, only 3 pairs (t-1, t) remain; this is a heuristic minimum for
# calculating the correlation, NOT a guarantee of statistical precision. See README.md.
_MIN_N_FOR_AUTOCORR = 4


@dataclass
class TrendResult:
    """Contains the statistical summary of a trend."""

    n: int
    sen_slope_decade: float       # units per decade
    ci_low_decade: float          # lower bound, units per decade
    ci_high_decade: float         # upper bound, units per decade
    p_value: float
    z: float
    trend: str                    # "increasing" | "decreasing" | "no trend"
    significant: bool
    lag1_autocorr: float
    method: str                   # "Mann-Kendall" or "Hamed-Rao"
    warnings: list[str] = field(default_factory=list)


def _calculate_lag1_autocorrelation(values: np.ndarray) -> float:
    """Calculate corr(values[:-1], values[1:]); return NaN if not possible."""
    if values.size < _MIN_N_FOR_AUTOCORR:
        return float("nan")

    previous = values[:-1]
    following = values[1:]

    # Pearson correlation is undefined if either array is constant.
    if np.std(previous) == 0 or np.std(following) == 0:
        return float("nan")

    correlation = np.corrcoef(previous, following)[0, 1]
    return float(correlation) if np.isfinite(correlation) else float("nan")


def trend_report(years, values, alpha: float = 0.05) -> TrendResult:
    """Analyze a monotonic trend in an annual time series.

    Parameters
    ----------
    years:
        Years of the observations. Some years may be missing.
    values:
        Observed values for each year. NaN values are allowed.
    alpha:
        Mann-Kendall significance level, between 0 and 1 (default 0.05).

    Returns
    -------
    TrendResult
        Sen's slope and confidence interval per decade, test results,
        estimated autocorrelation, method used, and warnings.

    Raises
    ------
    ValueError
        If alpha is not between 0 and 1, the inputs are not vectors of equal
        length, years are duplicated, or fewer than two valid observations remain.
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

    # Remove pairs where the year or value is not finite.
    valid_mask = np.isfinite(years_array) & np.isfinite(values_array)
    years_array = years_array[valid_mask]
    values_array = values_array[valid_mask]

    n = int(values_array.size)
    if n < 2:
        raise ValueError(
            "Se necesitan al menos dos observaciones válidas para calcular la pendiente."
        )

    # Sort both series using the same index to preserve year-value pairs.
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

    # Warn about gaps: Mann-Kendall tests receive the sequence of values,
    # while theilslopes does use the actual distances between years.
    if np.any(np.diff(years_array) != 1):
        warning_messages.append(
            "Hay huecos en years: la autocorrelación y Mann-Kendall usan observaciones "
            "válidas consecutivas, que pueden no corresponder a años calendario consecutivos."
        )

    # SciPy interprets alpha as the confidence level (e.g. 0.95), while
    # this function uses alpha as the significance level (e.g. 0.05).
    confidence_level = 1.0 - alpha
    slope, intercept, ci_low, ci_high = theilslopes(
        values_array,
        years_array,
        alpha=confidence_level,
    )

    # theilslopes returns units per year; multiplying by 10 gives units per decade.
    sen_slope_decade = float(slope * 10.0)
    ci_low_decade = float(ci_low * 10.0)
    ci_high_decade = float(ci_high * 10.0)

    # Remove the estimated trend before measuring autocorrelation. Calculating it
    # directly on values_array can produce a high correlation from a sustained trend,
    # even when the noise/residual has no temporal memory.
    fitted_trend = slope * years_array + intercept
    residuals = values_array - fitted_trend
    lag1_autocorr = _calculate_lag1_autocorrelation(residuals)
    if not np.isfinite(lag1_autocorr):
        warning_messages.append(
            "No fue posible estimar la autocorrelación lag-1 residual de forma fiable "
            "(se requieren al menos 4 observaciones y variación en los residuos); "
            "se utilizará Mann-Kendall original."
        )

    # Select the test according to the practical rule specified in the exercise.
    if (
        np.isfinite(lag1_autocorr)
        and abs(lag1_autocorr) > _AUTOCORR_THRESHOLD
    ):
        test_result = mk.hamed_rao_modification_test(values_array, alpha=alpha)
        method = "Hamed-Rao"
    else:
        test_result = mk.original_test(values_array, alpha=alpha)
        method = "Mann-Kendall"

    # Hamed-Rao can return NaN (e.g. with negative autocorrelation, the
    # corrected variance becomes negative). If this happens, use the original
    # Mann-Kendall test and issue a warning instead of reporting "not significant"
    # without explanation.
    if method == "Hamed-Rao" and not (
        np.isfinite(test_result.p) and np.isfinite(test_result.z)
    ):
        warning_messages.append(
            "Hamed-Rao no produjo un valor p válido; "
            "se utilizó Mann-Kendall original."
        )
        test_result = mk.original_test(values_array, alpha=alpha)
        method = "Mann-Kendall"

    p_value = float(test_result.p)
    z = float(test_result.z)
    trend = str(test_result.trend)

    if not (np.isfinite(p_value) and np.isfinite(z)):
        raise ValueError("No se pudo calcular un valor p válido para esta serie.")
    significant = bool(p_value < alpha)

    # Warn when the result is statistically significant, but the estimated change
    # over the entire period is small relative to the observed variability.
    # The relevant variability for this warning is the dispersion around the trend,
    # so we use the residuals rather than the original series.
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