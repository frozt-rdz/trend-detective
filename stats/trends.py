# AI Generated
"""Sen slopes on calendar years and rank-based tests on ordered observations."""

from dataclasses import dataclass, field

import numpy as np
from scipy.stats import theilslopes
import pymannkendall as mk


_AUTOCORR_THRESHOLD = 0.2
_SMALL_EFFECT_RATIO = 0.1
_MIN_N_FOR_AUTOCORR = 4


@dataclass
class TrendResult:
    """Primary inference and descriptive slope estimates for one series."""

    n: int
    sen_slope_decade: float
    ci_low_decade: float
    ci_high_decade: float
    p_value: float
    z: float
    trend: str
    significant: bool | None
    lag1_autocorr: float
    method: str
    warnings: list[str] = field(default_factory=list)
    status: str = "valid"
    test_variance: float = float("nan")


def _calculate_lag1_autocorrelation(values: np.ndarray) -> float:
    """Estimate Pearson correlation between adjacent residuals."""
    if values.size < _MIN_N_FOR_AUTOCORR:
        return float("nan")

    previous = values[:-1]
    following = values[1:]

    if np.std(previous) == 0 or np.std(following) == 0:
        return float("nan")

    correlation = np.corrcoef(previous, following)[0, 1]
    return float(correlation) if np.isfinite(correlation) else float("nan")


def trend_report(years, values, alpha: float = 0.05) -> TrendResult:
    """Estimate a decadal slope and select a test using detrended lag-one correlation."""
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

    valid_mask = np.isfinite(years_array) & np.isfinite(values_array)
    years_array = years_array[valid_mask]
    values_array = values_array[valid_mask]

    n = int(values_array.size)
    if n < 2:
        raise ValueError(
            "Se necesitan al menos dos observaciones válidas para calcular la pendiente."
        )

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

    if np.any(np.diff(years_array) != 1):
        warning_messages.append(
            "Hay huecos en years: la autocorrelación y Mann-Kendall usan observaciones "
            "válidas consecutivas, que pueden no corresponder a años calendario consecutivos."
        )

    confidence_level = 1.0 - alpha
    slope, intercept, ci_low, ci_high = theilslopes(
        values_array,
        years_array,
        alpha=confidence_level,
    )

    sen_slope_decade = float(slope * 10.0)
    ci_low_decade = float(ci_low * 10.0)
    ci_high_decade = float(ci_high * 10.0)

    fitted_trend = slope * years_array + intercept
    residuals = values_array - fitted_trend
    lag1_autocorr = _calculate_lag1_autocorrelation(residuals)
    if not np.isfinite(lag1_autocorr):
        warning_messages.append(
            "No fue posible estimar la autocorrelación lag-1 residual de forma fiable "
            "(se requieren al menos 4 observaciones y variación en los residuos); "
            "se utilizará Mann-Kendall original."
        )

    if (
        np.isfinite(lag1_autocorr)
        and abs(lag1_autocorr) > _AUTOCORR_THRESHOLD
    ):
        # Inspect the returned variance explicitly, including when the score is zero.
        with np.errstate(invalid="ignore", divide="ignore"):
            test_result = mk.hamed_rao_modification_test(values_array, alpha=alpha)
        method = "Hamed-Rao"
    else:
        test_result = mk.original_test(values_array, alpha=alpha)
        method = "Mann-Kendall"

    p_value = float(test_result.p)
    z = float(test_result.z)
    trend = str(test_result.trend)
    variance = float(test_result.var_s)
    status = "valid"
    significant = bool(p_value < alpha)
    if (not 0 <= p_value <= 1 or not np.isfinite(z)
            or not np.isfinite(variance) or (method == "Hamed-Rao" and variance <= 0)):
        status = "invalid_primary"
        significant = None
        trend = "unresolved"
        p_value = z = float("nan")
        warning_messages.append(
            f"{method} inference unavailable (score variance {variance:.6g})."
        )

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
        status=status,
        test_variance=variance,
    )
