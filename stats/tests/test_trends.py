"""Regression tests for stats.trends.trend_report.

The series in this module are synthetic and use fixed seeds to make the
results reproducible. They do not represent real observations.
"""

import numpy as np
import pytest
import pymannkendall as mk
from scipy.stats import theilslopes

from stats.trends import trend_report


ALPHA = 0.05


def _generate_ar1(seed: int = 42, n: int = 100, phi: float = 0.7) -> np.ndarray:
    """Generate stationary AR(1) noise with no deterministic trend."""
    rng = np.random.default_rng(seed)
    innovations = rng.normal(loc=0.0, scale=1.0, size=n)
    values = np.empty(n, dtype=float)

    # Initialize using the stationary variance of the AR(1) process.
    values[0] = innovations[0] / np.sqrt(1.0 - phi**2)
    for i in range(1, n):
        values[i] = phi * values[i - 1] + innovations[i]

    return values


def test_clear_trend_is_significant_and_slope_matches_scipy():
    """A known trend with little noise should be detected as increasing."""
    years = np.arange(1990, 2020)
    rng = np.random.default_rng(17)
    known_slope = 1.8  # units per year; data created solely for this test.
    values = known_slope * (years - years[0]) + rng.normal(0.0, 0.15, size=years.size)

    result = trend_report(years, values, alpha=ALPHA)

    # scipy.stats.theilslopes interprets alpha as the confidence level,
    # while trend_report receives the significance level.
    expected = theilslopes(values, years, alpha=1.0 - ALPHA)

    assert result.trend == "increasing"
    assert result.significant is True
    assert result.sen_slope_decade == pytest.approx(expected.slope * 10.0)
    assert result.ci_low_decade == pytest.approx(expected.low_slope * 10.0)
    assert result.ci_high_decade == pytest.approx(expected.high_slope * 10.0)


def test_white_noise_matches_original_mann_kendall():
    """With white noise and low residual autocorrelation, the original MK test is used."""
    rng = np.random.default_rng(0)
    years = np.arange(1950, 2030)
    values = rng.normal(loc=0.0, scale=1.0, size=years.size)

    result = trend_report(years, values, alpha=ALPHA)
    expected = mk.original_test(values, alpha=ALPHA)

    # This fixed seed produces residual autocorrelation below the heuristic
    # threshold of 0.2, so the Hamed-Rao modification should not be activated.
    assert result.method == "Mann-Kendall"
    assert result.p_value == pytest.approx(expected.p)


def test_ar1_without_trend_selects_hamed_rao_and_compares_p_values():
    """AR(1) should activate Hamed-Rao; no assumption is made that p is not significant."""
    years = np.arange(1900, 2000)
    values = _generate_ar1(seed=42, n=years.size, phi=0.7)

    result = trend_report(years, values, alpha=ALPHA)
    expected_corrected = mk.hamed_rao_modification_test(values, alpha=ALPHA)
    expected_original = mk.original_test(values, alpha=ALPHA)

    assert result.method == "Hamed-Rao"
    assert result.lag1_autocorr > 0.2
    assert result.p_value == pytest.approx(expected_corrected.p)

    # With this fixed realization, the correction changes the p-value compared
    # with the original Mann-Kendall test. No claim is made about significance.
    assert result.p_value != pytest.approx(expected_original.p, abs=1e-12)


@pytest.mark.parametrize("alpha", [0.0, 1.0, -0.01, 1.01])
def test_alpha_out_of_range_raises_value_error(alpha):
    years = np.arange(2000, 2006)
    values = np.arange(6, dtype=float)

    with pytest.raises(ValueError, match="alpha"):
        trend_report(years, values, alpha=alpha)


def test_duplicate_years_raise_value_error():
    years = np.array([2000, 2001, 2001, 2003])
    values = np.array([1.0, 2.0, 3.0, 4.0])

    with pytest.raises(ValueError, match="duplicate years"):
        trend_report(years, values)


def test_nan_is_removed_and_n_reflects_valid_observations():
    """Pairs with NaN are removed, and n reports how many remain."""
    years = np.arange(2000, 2010)
    rng = np.random.default_rng(23)
    values = 0.4 * (years - years[0]) + rng.normal(0.0, 0.3, size=years.size)
    values[[2, 7]] = np.nan

    result = trend_report(years, values, alpha=ALPHA)

    assert result.n == 8
    valid_pairs = np.isfinite(years) & np.isfinite(values)
    assert result.n == int(valid_pairs.sum())
    assert any("fewer than 10" in warning for warning in result.warnings)

def test_p_value_is_always_finite_with_noise():
    rng = np.random.default_rng(0)
    years = np.arange(2000, 2020)
    for _ in range(100):
        r = trend_report(years, rng.normal(size=20))
        assert np.isfinite(r.p_value) and np.isfinite(r.z)

def test_negative_autocorrelation_does_not_use_hamed_rao():
    # With negative autocorrelation Hamed-Rao can inflate z or give a negative
    # variance; only positive autocorrelation should trigger it.
    years = np.arange(2000, 2020)
    values = np.array([1.0, -1.0] * 10) + 0.01 * np.arange(20)
    result = trend_report(years, values)
    assert result.lag1_autocorr < -0.2
    assert result.method == "Mann-Kendall"
    assert result.status == "valid" and np.isfinite(result.p_value)
