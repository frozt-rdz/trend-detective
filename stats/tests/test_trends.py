"""Pruebas de regresión para stats.trends.trend_report.

Las series de este módulo son sintéticas y usan semillas fijas para que los
resultados sean reproducibles. No representan observaciones reales.
"""

import numpy as np
import pytest
import pymannkendall as mk
from scipy.stats import theilslopes

from stats.trends import trend_report


ALPHA = 0.05


def _generate_ar1(seed: int = 42, n: int = 100, phi: float = 0.7) -> np.ndarray:
    """Genera ruido AR(1) estacionario, sin tendencia determinista."""
    rng = np.random.default_rng(seed)
    innovations = rng.normal(loc=0.0, scale=1.0, size=n)
    values = np.empty(n, dtype=float)

    # Inicialización con la varianza estacionaria del proceso AR(1).
    values[0] = innovations[0] / np.sqrt(1.0 - phi**2)
    for i in range(1, n):
        values[i] = phi * values[i - 1] + innovations[i]

    return values


def test_tendencia_clara_es_significativa_y_pendiente_coincide_con_scipy():
    """Una tendencia conocida y poco ruido debe detectarse como creciente."""
    years = np.arange(1990, 2020)
    rng = np.random.default_rng(17)
    known_slope = 1.8  # unidades por año; datos creados únicamente para esta prueba.
    values = known_slope * (years - years[0]) + rng.normal(0.0, 0.15, size=years.size)

    result = trend_report(years, values, alpha=ALPHA)

    # scipy.stats.theilslopes interpreta alpha como nivel de confianza,
    # mientras que trend_report recibe el nivel de significancia.
    expected = theilslopes(values, years, alpha=1.0 - ALPHA)

    assert result.trend == "increasing"
    assert result.significant is True
    assert result.sen_slope_decade == pytest.approx(expected.slope * 10.0)
    assert result.ci_low_decade == pytest.approx(expected.low_slope * 10.0)
    assert result.ci_high_decade == pytest.approx(expected.high_slope * 10.0)


def test_ruido_blanco_coincide_con_mann_kendall_original():
    """Con ruido blanco y autocorrelación residual baja se usa MK original."""
    rng = np.random.default_rng(0)
    years = np.arange(1950, 2030)
    values = rng.normal(loc=0.0, scale=1.0, size=years.size)

    result = trend_report(years, values, alpha=ALPHA)
    expected = mk.original_test(values, alpha=ALPHA)

    # Esta semilla fija produce autocorrelación residual menor que el umbral
    # heurístico 0.2, así que no debe activarse la modificación Hamed-Rao.
    assert result.method == "Mann-Kendall"
    assert result.p_value == pytest.approx(expected.p)


def test_ar1_sin_tendencia_selecciona_hamed_rao_y_compara_p_values():
    """El AR(1) debe activar Hamed-Rao; no se presupone que p sea no significativo."""
    years = np.arange(1900, 2000)
    values = _generate_ar1(seed=42, n=years.size, phi=0.7)

    result = trend_report(years, values, alpha=ALPHA)
    expected_corrected = mk.hamed_rao_modification_test(values, alpha=ALPHA)
    expected_original = mk.original_test(values, alpha=ALPHA)

    assert result.method == "Hamed-Rao"
    assert result.lag1_autocorr > 0.2
    assert result.p_value == pytest.approx(expected_corrected.p)

    # Con esta realización fija, la corrección cambia el p-value respecto a
    # Mann-Kendall original. No se hace ninguna afirmación sobre significancia.
    assert result.p_value != pytest.approx(expected_original.p, abs=1e-12)


@pytest.mark.parametrize("alpha", [0.0, 1.0, -0.01, 1.01])
def test_alpha_fuera_de_rango_lanza_value_error(alpha):
    years = np.arange(2000, 2006)
    values = np.arange(6, dtype=float)

    with pytest.raises(ValueError, match="alpha"):
        trend_report(years, values, alpha=alpha)


def test_anios_duplicados_lanzan_value_error():
    years = np.array([2000, 2001, 2001, 2003])
    values = np.array([1.0, 2.0, 3.0, 4.0])

    with pytest.raises(ValueError, match="duplicados"):
        trend_report(years, values)


def test_nan_se_elimina_y_n_refleja_observaciones_validas():
    """Las parejas con NaN se eliminan y n informa cuántas quedan."""
    years = np.arange(2000, 2010)
    rng = np.random.default_rng(23)
    values = 0.4 * (years - years[0]) + rng.normal(0.0, 0.3, size=years.size)
    values[[2, 7]] = np.nan

    result = trend_report(years, values, alpha=ALPHA)

    assert result.n == 8
    valid_pairs = np.isfinite(years) & np.isfinite(values)
    assert result.n == int(valid_pairs.sum())
    assert any("menos de 10" in warning for warning in result.warnings)

def test_p_value_siempre_finito_en_ruido():
    rng = np.random.default_rng(0)
    anios = np.arange(2000, 2020)
    for _ in range(100):
        r = trend_report(anios, rng.normal(size=20))
        assert np.isfinite(r.p_value) and np.isfinite(r.z)