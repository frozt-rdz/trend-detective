# AI Generated
"""
Statistical analysis of NASA Giovanni environmental datasets.

Methods:
- Annual aggregation using complete calendar years
- Sen slope and confidence intervals
- Mann-Kendall / Hamed-Rao using stats.trends
- Benjamini-Hochberg FDR correction

Run:
    python -m stats.analyze_giovanni
"""

from pathlib import Path

import numpy as np
import pandas as pd

from pipeline.ingest_giovanni import REGIONS
from stats.trends import trend_report


PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_CSV = (
    PROJECT_ROOT / "data" / "processed"
    / "giovanni_observations.csv"
)

OUTPUT_CSV = (
    PROJECT_ROOT / "data" / "processed"
    / "giovanni_significance.csv"
)

ALPHA = 0.05

REQUIRED_COLUMNS = {
    "variable", "region", "date", "value",
    "units", "source", "doi"
}

EXPECTED_UNITS = {
    "precipitation_rate": "mm/day",
    "air_temperature_2m": "°C",
    "soil_moisture_0_10cm": "kg/m²",
}


def build_annual_series(
    monthly_data: pd.DataFrame,
    variable: str,
) -> tuple[pd.DataFrame, str]:
    """Aggregate observations using complete years only."""

    data = monthly_data.copy()

    data["date"] = pd.to_datetime(
        data["date"],
        errors="raise",
    )

    data["value"] = pd.to_numeric(
        data["value"],
        errors="raise",
    )

    expected_unit = EXPECTED_UNITS[variable]

    if set(data["units"].unique()) != {expected_unit}:
        raise ValueError(
            f"Unexpected units for {variable}"
        )

    if data["date"].dt.to_period("M").duplicated().any():
        raise ValueError("Duplicate monthly dates")

    if not data["date"].dt.is_month_start.all():
        raise ValueError("Expected first day of each month")

    data = data[
        np.isfinite(data["value"])
    ].copy()

    data["year"] = data["date"].dt.year
    data["month"] = data["date"].dt.month

    if variable == "precipitation_rate":
        # Convert monthly mean mm/day to monthly total mm.
        data["annual_component"] = (
            data["value"] * data["date"].dt.days_in_month
        )

        aggregation = "sum"
        annual_unit = "mm"

    else:
        data["annual_component"] = data["value"]

        aggregation = "mean"
        annual_unit = expected_unit

    annual = data.groupby("year").agg(
        annual_value=("annual_component", aggregation),
        month_count=("month", "nunique"),
    )

    annual = annual[
        annual["month_count"] == 12
    ].copy()

    return annual, annual_unit


def adjust_p_values(
    p_values: np.ndarray,
) -> np.ndarray:
    """BH on valid primary tests with the full planned family in the multiplier."""
    values = np.asarray(p_values, dtype=float)
    valid = np.isfinite(values)
    if np.any((values[valid] < 0) | (values[valid] > 1)):
        raise ValueError("Primary p-values must be probabilities")
    order = np.flatnonzero(valid)[np.argsort(values[valid])]
    ranked = values[order] * len(values) / np.arange(1, len(order) + 1)

    corrected = np.minimum.accumulate(
        ranked[::-1]
    )[::-1]

    corrected = np.clip(corrected, 0, 1)

    result = np.full(values.shape, np.nan)
    result[order] = corrected

    return result


def build_report(observations: pd.DataFrame) -> pd.DataFrame:
    """Evaluate the 18 planned comparisons over the 2000–2024 study window."""
    missing = REQUIRED_COLUMNS - set(observations.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    if (not observations.variable.isin(EXPECTED_UNITS).all()
            or not observations.region.isin(REGIONS).all()):
        raise ValueError("Unexpected variable or region in normalized observations")
    study_years = set(range(2000, 2025))
    records = []
    for variable, input_unit in EXPECTED_UNITS.items():
        for region in REGIONS:
            monthly = observations[
                (observations.variable == variable) & (observations.region == region)
            ]
            if monthly.empty:
                annual = pd.DataFrame(columns=["annual_value"])
                unit = "mm" if variable == "precipitation_rate" else input_unit
            else:
                annual, unit = build_annual_series(monthly, variable)
                if not pd.to_datetime(monthly.date).dt.year.isin(study_years).all():
                    raise ValueError("Observations outside the study window")
            years = annual.index.to_numpy(dtype=int)
            values = annual.annual_value.to_numpy(dtype=float)
            excluded = sorted(study_years.difference(years))
            warnings = []
            if excluded:
                warnings.append("Years without 12 valid months: " + ", ".join(map(str, excluded)))
            result = trend_report(years, values, alpha=ALPHA) if len(years) >= 4 else None
            if result:
                warnings.extend(result.warnings)
            else:
                warnings.append("At least four complete years are required for this analysis.")
            records.append({
                "variable": variable, "region": region, "units": unit,
                "source": monthly.source.iloc[0] if len(monthly) else None,
                "doi": monthly.doi.iloc[0] if len(monthly) else None,
                "start_year": years.min() if len(years) else None,
                "end_year": years.max() if len(years) else None,
                "n_years": len(years),
                "mean_annual_value": values.mean() if len(years) else np.nan,
                "sen_slope_decade": result.sen_slope_decade if result else np.nan,
                "ci_low_decade": result.ci_low_decade if result else np.nan,
                "ci_high_decade": result.ci_high_decade if result else np.nan,
                "p_value": result.p_value if result else np.nan,
                "trend": result.trend if result else "unresolved",
                "significant": result.significant if result else None,
                "lag1_autocorr": result.lag1_autocorr if result else np.nan,
                "test_variance": result.test_variance if result else np.nan,
                "method": result.method if result else "unavailable",
                "status": result.status if result else "insufficient_years",
                "warnings": " | ".join(warnings),
            })
    report = pd.DataFrame(records)
    report["q_value"] = adjust_p_values(report.p_value)
    report["significant_fdr"] = report.q_value.astype("Float64") <= ALPHA
    report["classification"] = report.significant_fdr.map({
        True: "significant", False: "not significant",
    }).fillna("unresolved")
    return report


def analyze_giovanni() -> pd.DataFrame:
    """Generate the report from normalized monthly observations."""
    report = build_report(pd.read_csv(INPUT_CSV))
    OUTPUT_CSV.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    report.to_csv(
        OUTPUT_CSV,
        index=False,
    )

    return report


def main() -> None:
    """Generate and display the statistical report."""

    report = analyze_giovanni()

    display_columns = [
        "variable",
        "region",
        "sen_slope_decade",
        "p_value",
        "q_value",
        "significant_fdr",
        "method",
    ]

    print("\nNASA GIOVANNI - TREND ANALYSIS")
    print("=" * 70)

    print(
        report[display_columns].to_string(
            index=False
        )
    )

    print("\nSUMMARY")
    print(f"Planned comparisons: {len(report)}; valid tests: {report.p_value.count()}")

    print(
        "Significant (p < 0.05): "
        f"{int(report['significant'].sum())}"
    )

    print(
        "Significant after FDR: "
        f"{int(report['significant_fdr'].sum())}"
    )

    print(f"\nOutput: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
