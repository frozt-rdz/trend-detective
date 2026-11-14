# AI-use notice: this file was written with the assistance of Claude (Anthropic)
# and reviewed by the team.
"""Trend Detective: is the Earth changing where you live? (Streamlit app)

Run from the repository root:
    streamlit run app/main.py

Data flow:  data/giovanni/*.csv --(pipeline.ingest_giovanni)--> giovanni_observations.csv
            --(pipeline.make_series_info)--> series_info.csv
            stats.analyze_giovanni builds annual series, trend tests and FDR correction.
"""

import sys
from math import cos, radians
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from stats.analyze_giovanni import ALPHA, INPUT_CSV, build_annual_series, build_report  # noqa: E402
from stats.trends import trend_report  # noqa: E402

INFO_CSV = ROOT / "data" / "processed" / "series_info.csv"

VARIABLES = {
    "precipitation_rate": {
        "label": "Precipitation",
        "plain": "How much rain falls. We add up the monthly rain to get a total in millimetres per year.",
    },
    "soil_moisture_0_10cm": {
        "label": "Soil moisture (top 10 cm)",
        "plain": "How wet the top 10 cm of soil is, in kilograms of water per square metre.",
    },
    "air_temperature_2m": {
        "label": "Air temperature (2 m)",
        "plain": "Temperature of the air two metres above the ground, in degrees Celsius.",
    },
}

REGIONS = {
    "sahel": "Sahel, West Africa",
    "south_amazon": "Southern Amazon, Brazil",
    "chiapas": "Chiapas (inland), Mexico",
    "morelos": "Morelos and surroundings, Mexico",
    "sonora": "Sonora (north-central, inland), Mexico",
    "zacatecas": "Zacatecas, Mexico",
}

VERDICT = {
    "significant": "A trend is detected",
    "not significant": "No trend can be claimed",
    "unresolved": "The test could not decide",
}

st.set_page_config(page_title="Trend Detective", page_icon="🌍", layout="wide")


@st.cache_data
def load_data(path: str, size: int, modified: int):
    """Read the monthly observations and compute the 18 planned trend tests."""
    monthly = pd.read_csv(path)
    return monthly, build_report(monthly)


@st.cache_data
def load_info(path: str, size: int, modified: int) -> pd.DataFrame:
    return pd.read_csv(path)


def fmt_p(p: float) -> str:
    if pd.isna(p):
        return "n/a"
    return "< 0.001" if p < 0.001 else f"{p:.3f}"


def fmt_slope(value: float, unit: str) -> str:
    return f"{value:+.3g} {unit}/decade"


def region_name(region: str) -> str:
    return REGIONS.get(region, region.replace("_", " ").title())


def box_caption(info_row: pd.Series) -> str:
    mid_lat = (info_row.box_north + info_row.box_south) / 2
    res_lat, res_lon = (float(x) for x in info_row.grid_resolution_deg_lat_lon.split("x"))
    cell_km = f"{res_lat * 111:.0f} x {res_lon * 111 * cos(radians(mid_lat)):.0f} km"
    return (
        f"Area analysed: {info_row.box_width_deg:g}° x {info_row.box_height_deg:g}° "
        f"(W {info_row.box_west:g}, S {info_row.box_south:g}, E {info_row.box_east:g}, "
        f"N {info_row.box_north:g}). About {int(info_row.approx_grid_cells)} grid cells "
        f"of roughly {cell_km} each. Values are averages over this box."
    )


# ---------------------------------------------------------------- data loading
if not INPUT_CSV.exists() or not INFO_CSV.exists():
    st.title("Trend Detective")
    st.error(
        "Data files are missing. From the repository root run: "
        "`python -m pipeline.ingest_giovanni` and `python -m pipeline.make_series_info`."
    )
    st.stop()

try:
    stat = INPUT_CSV.stat()
    monthly, report = load_data(str(INPUT_CSV), stat.st_size, stat.st_mtime_ns)
    stat_info = INFO_CSV.stat()
    info = load_info(str(INFO_CSV), stat_info.st_size, stat_info.st_mtime_ns)
except (ValueError, OSError, pd.errors.ParserError) as exc:
    st.error(f"The data could not be read: {exc}")
    st.stop()

# ---------------------------------------------------------------------- header
st.title("Trend Detective")
st.write(
    "**Is the Earth changing where you live?** Pick a variable and a region to see "
    "25 years (2000–2024) of NASA data, how much it changed, and whether the change is "
    "statistically significant."
)

variable = st.sidebar.selectbox(
    "Variable", list(VARIABLES), format_func=lambda v: VARIABLES[v]["label"]
)
regional = report[report.variable == variable].copy()
region = st.sidebar.selectbox(
    "Region", [r for r in REGIONS if r in set(regional.region)], format_func=region_name
)
st.sidebar.caption("Results cover the planned window 2000–2024 and years with all 12 months.")

row = regional[regional.region == region].iloc[0]
info_row = info[(info.variable == variable) & (info.region == region)].iloc[0]
unit = row.units

tab_explore, tab_compare, tab_about = st.tabs(["Explore a region", "Compare regions", "About the data and methods"])

# ------------------------------------------------------------------ tab: explore
with tab_explore:
    st.header(f"{VARIABLES[variable]['label']} in {region_name(region)}")
    st.write(VARIABLES[variable]["plain"])

    c1, c2, c3 = st.columns(3)
    c1.metric("Change per decade", fmt_slope(row.sen_slope_decade, unit))
    if pd.notna(row.p_value):
        c2.metric("Evidence (p-value)", fmt_p(row.p_value), help="Chance of a change this large if nothing were really changing.")
        c3.metric("After correcting for 18 comparisons (q)", fmt_p(row.q_value))
    else:
        c2.metric("Evidence (p-value)", "n/a")
        c3.metric("After correcting for 18 comparisons (q)", "n/a")

    verdict = VERDICT[row.classification]
    if row.classification == "significant":
        st.success(f"{verdict}: the {'increase' if row.sen_slope_decade > 0 else 'decrease'} is unlikely to be chance.")
    elif row.classification == "not significant":
        st.info(f"{verdict} with these data. This does not prove nothing is changing; 25 years is short.")
    else:
        st.warning(f"{verdict}.")
    if row.warnings:
        st.warning(row.warnings)

    st.write(
        f"Conventional 95% interval for the change: {row.ci_low_decade:+.3g} to "
        f"{row.ci_high_decade:+.3g} {unit}/decade. Complete years used: {row.n_years}. "
        f"Method: {row.method}."
    )

    series = monthly[(monthly.variable == variable) & (monthly.region == region)]
    annual, _ = build_annual_series(series, variable)
    figure = go.Figure()
    figure.add_scatter(x=annual.index, y=annual.annual_value, mode="markers", name="Yearly value")
    if np.isfinite(row.sen_slope_decade):
        years = annual.index.to_numpy(dtype=float)
        fitted = annual.annual_value.median() + (row.sen_slope_decade / 10) * (years - np.median(years))
        figure.add_scatter(x=annual.index, y=fitted, mode="lines", name="Trend (Sen's slope)")
    figure.update_layout(
        template="plotly_white",
        xaxis_title="Year",
        yaxis_title=f"{VARIABLES[variable]['label']} ({unit})",
        legend=dict(orientation="h", y=1.1),
    )
    st.plotly_chart(figure)
    st.caption(box_caption(info_row))

    with st.expander("Does the result hold if we start later (2005–2024)?"):
        later = annual[annual.index >= 2005]
        if len(later) >= 4:
            check = trend_report(later.index.to_numpy(), later.annual_value.to_numpy(), alpha=ALPHA)
            st.write(
                f"From 2005: {fmt_slope(check.sen_slope_decade, unit)}, p = {fmt_p(check.p_value)} "
                f"({check.method}, {check.n} years)."
            )
            same_sign = np.sign(check.sen_slope_decade) == np.sign(row.sen_slope_decade)
            still_significant = bool(check.significant)
            if not same_sign:
                st.write("The direction changes: the result is fragile.")
            elif still_significant:
                st.write("Same direction, and still significant at 0.05 (before any multiple-testing correction).")
            else:
                st.write("Same direction, but no longer significant at 0.05: the evidence depends on the start year.")
            st.caption(
                "This is a robustness check, not a new test: it is not part of the 18 planned "
                "comparisons and is not corrected for multiple testing."
            )
        else:
            st.write("Not enough complete years from 2005 for this series.")

    st.caption(
        f"{info_row.source} · {info_row.nature} · "
        f"[DOI: {info_row.doi}](https://doi.org/{info_row.doi})"
    )

# ----------------------------------------------------------------- tab: compare
with tab_compare:
    st.header(f"{VARIABLES[variable]['label']}: change per decade by region")

    rising = regional[(regional.classification == "significant") & (regional.sen_slope_decade > 0)]
    falling = regional[(regional.classification == "significant") & (regional.sen_slope_decade < 0)]
    if len(rising) and len(falling):
        st.success(
            "**Opposite trends:** rising in "
            + ", ".join(region_name(r) for r in rising.region)
            + "; falling in "
            + ", ".join(region_name(r) for r in falling.region)
            + "."
        )

    ordered = regional.sort_values("sen_slope_decade")
    names = ordered.region.map(region_name)
    compare = go.Figure(go.Scatter(
        x=ordered.sen_slope_decade, y=names, mode="markers",
        marker=dict(size=10, color=np.where(ordered.classification == "significant", "#1f77b4", "#9e9e9e")),
        error_x=dict(
            type="data", symmetric=False,
            array=ordered.ci_high_decade - ordered.sen_slope_decade,
            arrayminus=ordered.sen_slope_decade - ordered.ci_low_decade,
        ),
        name="Change per decade",
    ))
    compare.add_vline(x=0, line_dash="dot")
    compare.update_layout(template="plotly_white", xaxis_title=f"Change per decade ({unit}/decade)", showlegend=False)
    st.plotly_chart(compare)
    st.caption(
        "Blue dots: significant after correcting for 18 comparisons. Grey dots: not significant. "
        "Bars are conventional 95% intervals; they do not adjust for autocorrelation."
    )

    table = pd.DataFrame({
        "Region": ordered.region.map(region_name),
        "Change per decade": [fmt_slope(v, unit) for v in ordered.sen_slope_decade],
        "p": [fmt_p(v) for v in ordered.p_value],
        "q": [fmt_p(v) for v in ordered.q_value],
        "Verdict": ordered.classification.map(VERDICT),
    })
    st.dataframe(table, hide_index=True)

# ------------------------------------------------------------------- tab: about
with tab_about:
    st.header("The variables")
    for key, meta in VARIABLES.items():
        nature = info[info.variable == key].nature.iloc[0]
        st.markdown(f"**{meta['label']}.** {meta['plain']} _{nature}._")

    st.header("How to read the numbers")
    st.markdown(
        "- **Yearly value:** the average of the 12 monthly values (for rain, the sum of the year). "
        "Only years with all 12 months are used.\n"
        "- **Change per decade (Sen's slope):** the typical change every 10 years. It is robust to unusual years.\n"
        "- **p-value:** if nothing were really changing, how often would chance alone give a change this large? "
        "Below 0.05 means it would be unusual.\n"
        "- **q-value:** the p-value corrected for running 18 comparisons at once, so lucky results are not "
        "overstated (Benjamini-Hochberg).\n"
        "- **Method:** Mann-Kendall tests whether values tend to rise or fall. Hamed-Rao is used instead when "
        "consecutive years are positively correlated, which would otherwise make the test too confident."
    )

    st.header("Limitations")
    st.markdown(
        "- 25 years is short. Rainfall in Mexico also swings with El Niño and other multi-year cycles.\n"
        "- A result that is not significant does not prove that nothing is changing.\n"
        "- Boxes are averages over many grid cells; trends can differ inside a box. "
        "Boxes were chosen by the data, not by labels such as dry or rainy.\n"
        "- Temperature and soil moisture come from models fed with observations; precipitation is a satellite estimate.\n"
        "- Linear trends only; seasonal or non-linear changes are out of scope."
    )

    st.header("Data provenance")
    shown = info[["variable", "region", "source", "doi", "units", "approx_grid_cells", "valid_months"]].copy()
    shown["region"] = shown.region.map(region_name)
    st.dataframe(shown, hide_index=True)
    st.caption(
        "Source: NASA Giovanni time series (area-averaged, monthly). "
        "MERRA-2 (doi:10.5067/AP1B0BA5PD2K), GPM IMERG Final V07 (doi:10.5067/GPM/IMERG/3B-MONTH/07), "
        "GLDAS Noah 2.1 (doi:10.5067/SXAVCZFAQLNO). "
        "This project uses NASA open data; it is not endorsed by NASA."
    )
    st.caption(
        "Use of AI: code in this project was written with the assistance of Claude (Anthropic) "
        "and reviewed by the team. The project idea and the choice of regions are the team's own."
    )
