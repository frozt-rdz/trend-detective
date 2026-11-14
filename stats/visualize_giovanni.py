# AI Generated
"""Interactive interpretation of annual Giovanni trends."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# Streamlit can launch a script with only its directory on the import path.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from stats.analyze_giovanni import ALPHA, INPUT_CSV, build_annual_series, build_report

LABELS = {
    "precipitation_rate": "Annual precipitation",
    "air_temperature_2m": "2 m air temperature",
    "soil_moisture_0_10cm": "Soil moisture at 0–10 cm",
}


def region_label(region):
    return region.replace("_", " ").title()


@st.cache_data
def load_report(csv_bytes):
    from io import BytesIO

    monthly = pd.read_csv(BytesIO(csv_bytes))
    return monthly, build_report(monthly)


def main():
    st.set_page_config(page_title="Giovanni regional trends", layout="wide")
    st.title("Regional climate trends")
    st.caption("NASA Giovanni monthly area averages · 2000–2024")
    if not INPUT_CSV.exists():
        st.error("Generate the observations with python -m pipeline.ingest_giovanni.")
        return
    monthly, report = load_report(INPUT_CSV.read_bytes())
    variable = st.selectbox("Environmental variable", LABELS.keys(), format_func=LABELS.get)
    regional = report[report.variable == variable].copy()
    region = st.selectbox("Geographic region", regional.region, format_func=region_label)
    row = regional[regional.region == region].iloc[0]
    series = monthly[(monthly.variable == variable) & (monthly.region == region)]
    unit = row.units
    slope_unit = f"{unit}/decade"

    st.header(f"{LABELS[variable]} in {region_label(region)}")
    if row.warnings:
        st.warning(row.warnings)
    st.write(
        "Annual precipitation sums the monthly mean daily rate multiplied by calendar days in each month."
        if variable == "precipitation_rate" else
        "Annual values are the arithmetic mean of twelve monthly area averages."
    )
    if not series.empty:
        annual, _ = build_annual_series(series, variable)
        chart = go.Figure()
        chart.add_scatter(x=annual.index, y=annual.annual_value, mode="markers", name="Complete year")
        if np.isfinite(row.sen_slope_decade):
            slope = row.sen_slope_decade / 10
            fitted = annual.annual_value.median() + slope * (annual.index - np.median(annual.index))
            chart.add_scatter(x=annual.index, y=fitted, mode="lines", name="Sen trend")
        chart.update_layout(xaxis_title="Calendar year", yaxis_title=f"{LABELS[variable]} ({unit})")
        st.plotly_chart(chart)
    interpretation = {
        "significant": "The selected test detects a monotonic trend after FDR correction.",
        "not significant": "The evidence does not meet the FDR threshold for a monotonic trend.",
        "unresolved": "The primary test cannot establish significance. The slope describes the observed change.",
    }
    st.write(interpretation[row.classification])
    if np.isfinite(row.sen_slope_decade):
        st.write(f"Estimated change per decade: {row.sen_slope_decade:+.3g} {unit} "
                 f"(conventional 95% interval {row.ci_low_decade:.3g} to {row.ci_high_decade:.3g}).")
    st.write(f"Complete years: {row.n_years} · Method: {row.method} · Residual lag-one correlation: "
             + (f"{row.lag1_autocorr:.3f}" if pd.notna(row.lag1_autocorr) else "unavailable"))
    if pd.notna(row.p_value):
        st.write(f"Primary p = {row.p_value:.4g} · BH q = {row.q_value:.4g}")
    else:
        st.write("Primary p and BH q are unavailable.")
    if pd.notna(row.doi):
        st.caption(f"{row.source} · DOI {row.doi}")

    st.header("Change per decade across regions")
    labels = regional.region.map(region_label)
    chart = go.Figure(go.Scatter(
        x=regional.sen_slope_decade, y=labels, mode="markers", name="Sen slope",
        error_x=dict(type="data", symmetric=False,
                     array=regional.ci_high_decade - regional.sen_slope_decade,
                     arrayminus=regional.sen_slope_decade - regional.ci_low_decade),
    ))
    chart.add_vline(x=0, line_dash="dot")
    chart.update_xaxes(title=f"Sen slope ({slope_unit})")
    st.plotly_chart(chart)
    st.caption("Conventional 95% Sen intervals do not adjust for serial autocorrelation.")

    st.header("Evidence for a monotonic trend")
    categories = labels + " · " + regional.classification
    chart = go.Figure()
    for column, label, symbol in [("p_value", "Primary p", "circle"), ("q_value", "BH q", "diamond-open")]:
        chart.add_scatter(x=regional[column], y=categories, mode="markers", name=label,
                          marker=dict(symbol=symbol, size=10))
    chart.add_vline(x=ALPHA, line_dash="dash", annotation_text="0.05")
    chart.update_xaxes(title="Probability value", range=[-0.02, 1.02])
    chart.update_yaxes(categoryorder="array", categoryarray=categories, tickmode="array",
                       tickvals=categories, ticktext=categories)
    st.plotly_chart(chart)
    st.caption(f"{report.p_value.count()} valid primary tests across 18 planned comparisons. "
               "Unresolved regions have no p or q marker. No diagnostic fallback is used.")
    st.dataframe(regional[["region", "method", "p_value", "q_value", "classification"]], hide_index=True)


if __name__ == "__main__":
    main()
