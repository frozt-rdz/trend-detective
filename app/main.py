from pathlib import Path
import sys

import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.analisis import analizar_tendencia, formatear_p  # noqa: E402


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

# Decimales a mostrar por variable (las que no estén aquí usan 3).
DECIMALES = {"temperatura_superficial": 3, "humedad_suelo": 4}

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

inicio, fin = selected
if fin - inicio < 4:  #con menos años trend_report no es fiable
    st.warning("Elige un periodo de al menos 5 años.")
    st.stop()

unit = serie["unidades"].iloc[0]
# Pendiente lineal exploratoria (no sustituye a Sen/Mann-Kendall).
ap = DECIMALES.get(variable, 3)  # ecimales según la variable

# IMPORTANTE: analizar_tendencia recibe la serie COMPLETA (sin recortar por el slider),
# porque calcula el clima de referencia 2000–2019; el slider solo fija inicio y fin.
try:
    r, anual = analizar_tendencia(serie, inicio, fin)
except (ValueError, ZeroDivisionError) as exc:
    st.warning(f"No fue posible calcular el análisis estadístico: {exc}")
    st.stop()

st.subheader(f"{labels.get(variable, variable)} — {region}")
a, b, c, d = st.columns(4)
a.metric("Pendiente de Sen", f"{r.sen_slope_decade:+.{ap}f} {unit}/década")
b.metric("Valor p", formatear_p(r.p_value))
c.metric("Método", r.method)
d.metric("Último registro", f"{serie['valor'].iloc[-1]:.3f} {unit}")
st.write(f"IC 95 %: [{r.ci_low_decade:+.{ap}f}, {r.ci_high_decade:+.{ap}f}] {unit}/década · {r.n} años completos")
if not r.significant:
    st.info("No se puede afirmar una tendencia con estos datos.")
for aviso in r.warnings:
    st.warning(aviso)

# Anomalía anual + recta de Sen (la misma pendiente que se reporta arriba).
anios = anual.index.to_series()
sen_linea = anual.median() + (r.sen_slope_decade / 10) * (anios - anios.median())
grafica = pd.DataFrame({"Anomalía anual": anual, "Tendencia (Sen)": sen_linea})
grafica.index = pd.to_datetime(grafica.index.astype(str), format="%Y")
st.line_chart(grafica, y_label=f"Anomalía ({unit})", x_label="Año")
st.caption("Anomalía respecto al promedio mensual 2000–2019. Solo se analizan años con 12 meses disponibles.")

with st.expander("Ver serie mensual original"):
    mensual = serie[serie["fecha"].dt.year.between(inicio, fin)]
    st.line_chart(mensual.set_index("fecha")["valor"], y_label=unit)


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

# Comparación de regiones: solo tiene sentido si la variable tiene 2 o más regiones.
regiones = sorted(subset["region"].unique())
if len(regiones) > 1:
    filas = []
    for reg in regiones:
        serie_reg = subset[subset["region"] == reg]
        try:
            res, _ = analizar_tendencia(serie_reg, inicio, fin)
        except (ValueError, ZeroDivisionError):
            filas.append({"Región": reg, "Pendiente": "No disponible", "IC 95 %": "—", "p": "—", "¿Significativa?": "—"})
            continue
        filas.append({
            "Región": reg,
            "Pendiente": f"{res.sen_slope_decade:+.{ap}f} {unit}/década",
            "IC 95 %": f"[{res.ci_low_decade:+.{ap}f}, {res.ci_high_decade:+.{ap}f}]",
            "p": formatear_p(res.p_value),
            "¿Significativa?": "Sí" if res.significant else "No",
        })
    st.subheader("Comparación de regiones")
    st.dataframe(pd.DataFrame(filas), hide_index=True)