"""Smoke tests for the Streamlit app and the series_info provenance table."""

import json
from pathlib import Path

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from pipeline.make_series_info import CATALOG, build_series_info, clean_url, grid_cells

ROOT = Path(__file__).resolve().parents[2]
APP = ROOT / "app" / "main.py"


def test_series_info_covers_the_18_series_and_cleans_urls():
    info = build_series_info(json.loads(CATALOG.read_text(encoding="utf-8")))
    assert len(info) == 18 and not info.duplicated(["variable", "region"]).any()
    assert (info.valid_months == 300).all()
    assert not info.giovanni_url.str.contains("utm_source").any()
    assert info.doi.str.startswith("10.5067/").all()


def test_clean_url_and_grid_cells():
    assert clean_url("https://x.org/giovanni/?utm_source=chatgpt.com#service=ArAvTs") == \
        "https://x.org/giovanni/#service=ArAvTs"
    # MERRA-2 box: 4 longitudes x 6 latitudes of cell centres
    assert grid_cells([-111.25, 29.0, -109.375, 31.5], [0.5, 0.625]) == 24


def test_app_runs_for_every_variable_and_region():
    app = AppTest.from_file(str(APP), default_timeout=90).run()
    assert not app.exception
    for variable in ["precipitation_rate", "soil_moisture_0_10cm", "air_temperature_2m"]:
        app.sidebar.selectbox[0].set_value(variable).run()
        assert not app.exception
        for region in ["sahel", "south_amazon", "chiapas", "morelos", "sonora", "zacatecas"]:
            app.sidebar.selectbox[1].set_value(region).run()
            assert not app.exception


def test_app_shows_opposite_precipitation_trends():
    app = AppTest.from_file(str(APP), default_timeout=90).run()
    app.sidebar.selectbox[0].set_value("precipitation_rate").run()
    messages = [item.value for item in app.success]
    assert any("Opposite trends" in m and "Sahel" in m and "Amazon" in m for m in messages)
