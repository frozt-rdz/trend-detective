from pathlib import Path
import sys

import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from stats.trends import trend_report  # noqa: E402

DATA_PATH = ROOT / "data" / "processed" / "observaciones.csv"
SAMPLE_PATH = ROOT / "data" / "sample" / "muestra_sintetica.csv"
REQUIRED = {"variable", "region", "fecha", "valor", "unidades", "fuente", "doi"}

st.set_page_config(page_title="Detective de tendencias", page_icon="🌍", layout="wide")
st.title("Trend Detective")

# Modo explícito: nunca presentar una muestra sintética como dato científico.
options = ["Datos reales"]
if SAMPLE_PATH.exists():
    options.append("Demostración sintética")
modo = st.sidebar.radio("Fuente de datos", options)
source_path = DATA_PATH if modo == "Datos reales" else SAMPLE_PATH

@st.cache_data
# El path, tamaño y fecha de modificación invalidan la caché cuando cambia el CSV.
def cargar(path: str, size: int, modified: int) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["fecha"])
    missing = REQUIRED - set(df.columns)
    if missing:
        raise ValueError(f"Faltan columnas: {sorted(missing)}")
    df["valor"] = pd.to_numeric(df["valor"], errors="coerce")
    return df.dropna(subset=["fecha", "valor"])

if not source_path.exists():
    st.info(
        "Todavía no existe el CSV real. Desde la raíz del repositorio ejecuta: "
        "`python pipeline/ingest_nsidc_gistemp.py`"
    )
    st.stop()

try:
    stat = source_path.stat()
    df = cargar(str(source_path), stat.st_size, stat.st_mtime_ns)
except (ValueError, OSError, pd.errors.ParserError) as exc:
    st.error(f"No se pudieron leer los datos: {exc}")
    st.stop()

if df.empty:
    st.warning("No hay observaciones válidas.")
    st.stop()

if modo != "Datos reales":
    st.warning("Estos datos son SINTÉTICOS y no deben citarse como evidencia científica.")

labels = {
    "extension_hielo_marino": "Extensión del hielo marino",
    "anomalia_temperatura": "Anomalía de temperatura",
}
variables = sorted(df["variable"].unique())
variable = st.sidebar.selectbox("Variable", variables, format_func=lambda v: labels.get(v, v))
subset = df[df["variable"] == variable].copy()
region = st.sidebar.selectbox("Región", sorted(subset["region"].unique()))
serie = subset[subset["region"] == region].sort_values("fecha").copy()

min_year, max_year = int(serie["fecha"].dt.year.min()), int(serie["fecha"].dt.year.max())
if min_year == max_year:
    st.sidebar.caption(f"Periodo disponible: {min_year}")
    selected = (min_year, max_year)
else:
    selected = st.sidebar.slider("Periodo", min_year, max_year, (min_year, max_year))

serie = serie[serie["fecha"].dt.year.between(*selected)].copy()
if len(serie) < 2:
    st.warning("Necesitamos al menos 2 observaciones para mostrar una tendencia.")
    st.stop()

unit = serie["unidades"].iloc[0]
# Pendiente lineal exploratoria (no sustituye a Sen/Mann-Kendall).
x = serie["fecha"].dt.year + (serie["fecha"].dt.month - 1) / 12
slope, intercept = np.polyfit(x.to_numpy(float), serie["valor"].to_numpy(float), 1)
serie["tendencia_lineal"] = intercept + slope * x

col1, col2, col3 = st.columns(3)
col1.metric("Observaciones mensuales", len(serie))
col2.metric("Último registro", f"{serie['valor'].iloc[-1]:.3f} {unit}")
col3.metric("Pendiente lineal exploratoria", f"{10 * slope:+.3f} {unit}/década")

st.subheader(f"{labels.get(variable, variable)} — {region}")
st.line_chart(serie.set_index("fecha")[["valor", "tendencia_lineal"]])


# Mostrar la procedencia de los datos debajo de la gráfica.
if modo == "Demostración sintética":
    st.caption(
        "Datos sintéticos utilizados únicamente para pruebas. "
        "No representan observaciones científicas reales."
    )

elif variable == "extension_hielo_marino":
    st.caption(
        "Fuente: NSIDC Sea Ice Index v4 (G02135) · "
        "[DOI: 10.7265/a98x-0f50]"
        "(https://doi.org/10.7265/a98x-0f50)"
    )

elif variable == "anomalia_temperatura":
    st.caption(
        "Fuente: NASA GISS GISTEMP v4 · "
        "Base climatológica: 1951–1980 · "
        "[Datos oficiales]"
        "(https://data.giss.nasa.gov/gistemp/)"
    )

else:
    st.caption(f"Fuente: {serie['fuente'].iloc[0]}")

if variable == "anomalia_temperatura":
    st.info(
        "La anomalía se mide respecto al promedio de "
        "1951–1980; no representa temperatura absoluta."
    )

# El módulo estadístico recibe solamente AÑOS COMPLETOS y una media anual por año.
serie["anio"] = serie["fecha"].dt.year
annual = serie.groupby("anio").agg(
    media=("valor", "mean"),
    meses=("fecha", "nunique"),
)
annual = annual[annual["meses"] == 12]
st.subheader("Análisis estadístico sobre años completos")
if len(annual) < 4:
    st.warning("Se necesitan más años completos para interpretar la tendencia con cautela.")
else:
    try:
        result = trend_report(annual.index.to_numpy(), annual["media"].to_numpy())
        a, b, c = st.columns(3)
        a.metric("Pendiente de Sen", f"{result.sen_slope_decade:+.3f} {unit}/década")
        b.metric("Valor p", f"{result.p_value:.4g}")
        c.metric("Método", result.method)
        st.write(f"Tendencia: **{result.trend}**. Significativa: **{'Sí' if result.significant else 'No'}**.")
        st.caption(f"Intervalo de confianza: [{result.ci_low_decade:.3f}, {result.ci_high_decade:.3f}] {unit}/década")
        for warning in result.warnings:
            st.warning(warning)
    except (ValueError, ZeroDivisionError) as exc:
        st.warning(f"No fue posible calcular el análisis estadístico: {exc}")

st.caption("Los datos de 2026 pueden ser parciales. Solo se analizan años con 12 meses disponibles.")
