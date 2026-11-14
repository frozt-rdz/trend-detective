# AI Generated
"""Checks against original exports and independently known aggregation results."""

from pathlib import Path
from shutil import copyfile

import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from pipeline.ingest_giovanni import (
    INPUT_DIR, GiovanniValidationError, build_giovanni_dataset, read_giovanni_csv,
)
from stats.analyze_giovanni import adjust_p_values, build_annual_series, build_report


@pytest.mark.parametrize(("filename", "region"), [
    ("soil_chiapas.csv", "chiapas"), ("soil_south_amazon.csv", "south_amazon"),
    ("suelo_suramazonias.csv", "south_amazon"),
])
def test_canonical_names_and_contract(tmp_path, filename, region):
    path = tmp_path / filename
    copyfile(INPUT_DIR / f"soil_{region}.csv", path)
    frame, entry = read_giovanni_csv(path)
    assert list(frame) == ["variable", "region", "date", "value", "units", "source", "doi"]
    assert frame.region.unique().tolist() == [region]
    assert entry["region"] == region
    assert entry["doi"] == "10.5067/SXAVCZFAQLNO"


def test_duplicate_canonical_series(tmp_path):
    for filename in ["soil_south_amazon.csv", "suelo_suramazonias.csv"]:
        copyfile(INPUT_DIR / "soil_south_amazon.csv", tmp_path / filename)
    with pytest.raises(GiovanniValidationError, match="Duplicate canonical series"):
        build_giovanni_dataset(tmp_path)


@pytest.mark.parametrize(("variable", "unit"), [
    ("precipitation_rate", "mm/day"), ("air_temperature_2m", "°C"),
    ("soil_moisture_0_10cm", "kg/m²"),
])
def test_calendar_aggregation(variable, unit):
    frame = pd.DataFrame(dict(date=pd.date_range("2000-01-01", periods=36, freq="MS"),
                              value=1., units=unit))
    frame = frame.drop(30)
    annual, annual_unit = build_annual_series(frame, variable)
    assert list(annual.index) == [2000, 2001]
    if variable == "precipitation_rate":
        assert list(annual.annual_value) == [366, 365]
        assert annual_unit == "mm"
    else:
        assert list(annual.annual_value) == [1, 1]
        assert annual_unit == unit
    with pytest.raises(ValueError, match="Duplicate monthly"):
        build_annual_series(pd.concat([frame, frame.iloc[:1]]), variable)


def test_planned_family_bh():
    result = adjust_p_values(np.array([0.04, np.nan, 0.01, 0.025]))
    np.testing.assert_allclose(result, [0.04 * 4 / 3, np.nan, 0.04, 0.05])


def test_real_sahel_regression_and_determinism():
    monthly, catalog = build_giovanni_dataset()
    again, second_catalog = build_giovanni_dataset()
    pd.testing.assert_frame_equal(monthly, again)
    assert catalog == second_catalog
    assert set(monthly.doi) == {
        "10.5067/GPM/IMERG/3B-MONTH/07", "10.5067/AP1B0BA5PD2K", "10.5067/SXAVCZFAQLNO",
    }
    report = build_report(monthly)
    assert len(report) == 18 and report.q_value.count() == 18
    # Sahel precipitation has NEGATIVE lag-1 autocorrelation: Hamed-Rao is not used
    # (its correction is meant for positive autocorrelation), so original
    # Mann-Kendall gives a valid, interpretable result.
    row = report.query("variable == 'precipitation_rate' and region == 'sahel'").iloc[0]
    assert row.n_years == 25 and row.method == "Mann-Kendall"
    assert row.lag1_autocorr == pytest.approx(-0.5685852441)
    assert row.status == "valid" and row.p_value == pytest.approx(0.012455, abs=1e-5)
    assert row.q_value == pytest.approx(0.028, abs=1e-3) and bool(row.significant_fdr)
    assert row.classification == "significant"
    assert row.sen_slope_decade == pytest.approx(41.700279)
    # Opposite signs for precipitation: Sahel up, South Amazon down, both FDR-significant.
    amazon = report.query("variable == 'precipitation_rate' and region == 'south_amazon'").iloc[0]
    assert row.sen_slope_decade > 0 > amazon.sen_slope_decade and bool(amazon.significant_fdr)
    absent = build_report(monthly[monthly.region != "sahel"]).query("region == 'sahel'")
    assert set(absent.status) == {"insufficient_years"}
    assert absent.significant_fdr.isna().all()


def test_dashboard_sahel():
    script = Path(__file__).resolve().parents[1] / "visualize_giovanni.py"
    app = AppTest.from_file(str(script), default_timeout=30).run()
    assert not app.exception
    app.selectbox[1].set_value("sahel").run()
    assert not app.exception
    assert not any("unavailable" in item.value for item in app.warning)
    assert len(app.get("plotly_chart")) == 3
