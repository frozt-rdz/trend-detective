from pathlib import Path
import pandas as pd
import streamlit as st
import numpy as np
import sys

RAIZ = Path(__file__).parent.parent
sys.path.insert(0, str(RAIZ))                 #para poder importar stats y app
from app.analisis import analizar_tendencia, formatear_p

st.set_page_config(page_title="Detective de tendencias", layout="wide")

@st.cache_data #streamlit recarga todo el codigo, con esto guarda los resultados del CSV y no los relee con cada clic
def cargar():
    ruta = RAIZ / "data" / "sample" / "muestra_sintetica.csv" #ruta de los datos
    return pd.read_csv(ruta, parse_dates=["fecha"]) #pasa las fechas a algo digerible

df = cargar() #llamamos a la funcion

col1, col2 = st.columns(2) #añadimos a la app dos columnas

variable = col1.selectbox("Variable", sorted(df['variable'].unique()))
region = col2.selectbox("Region", sorted(df['region'].unique()))

serie = df[(df['region'] == region) & (df['variable'] == variable)].dropna(subset=["valor"]).sort_values("fecha")
#serie de datos con la que se trabajara

unidad = serie['unidades'].iloc[0] #sale de los datos

DECIMALES = {"temperatura_superficial" : 3, "humedad_suelo": 4} #diccionario para el num de decimales por cada variable

anio_min = int(df['fecha'].dt.year.min()) #agregamos columnas a la serie para manejar año y mes
anio_max = int(df['fecha'].dt.year.max())
inicio, fin = st.slider("Periodo", anio_min, anio_max, (anio_min, anio_max)) #slider para elegir año

if fin - inicio < 4:                  #con menos años la función se cae
    st.warning("Elige un periodo de al menos 5 años.")
    st.stop()                         #detiene el script: nada de abajo se ejecuta

r, anual = analizar_tendencia(serie, inicio, fin)

ap = DECIMALES[variable]
st.metric("Pendiente de Sen", f"{r.sen_slope_decade:+.{ap}f} {unidad}/década") #mostramos la pendiente de sen
st.write(f"IC [{r.ci_low_decade:+.{ap}f}, {r.ci_high_decade:+.{ap}f}], {formatear_p(r.p_value)}, Método: {r.method}") #datos
if not r.significant:
    st.info("No se puede afirmar una tendencia con estos datos.")

for aviso in r.warnings: #imprimimos los avisos de tipo warning si es que existen
    st.warning(aviso)

anios = anual.index.to_series()      #los años
sen_linea = anual.median() + (r.sen_slope_decade / 10) * (anios - anios.median())   #mediana de anual + pendiente por AÑO * (año - mediana de los años)
grafica = pd.DataFrame({"Anomalía anual": anual, "Tendencia (Sen)": sen_linea})
grafica.index = pd.to_datetime(grafica.index.astype(str), format="%Y") #para que los años no se vean como 2,000
st.line_chart(grafica, y_label=f"Anomalía ({unidad})", x_label="Año") #indices de la grafica

#tabla para comparar regiones
def evaluar_sig(x):
    if not x.significant:
        return "No"
    else:
        return "Sí"
    
filas = []
for reg in sorted(df["region"].unique()): #recorremos segun las regiones que existan
    serie_reg = df[(df["variable"] == variable) & (df["region"] == reg)] #serie dependiendo variable y region
    res, _ = analizar_tendencia(serie_reg, inicio, fin)   #"_" no se necesita la serie anual aquí
    filas.append({
        "Región": reg,
        "Pendiente": f"{res.sen_slope_decade:+.{ap}f} {unidad}/década",  
        "IC 95 %": f"[{res.ci_low_decade:+.{ap}f}, {res.ci_high_decade:+.{ap}f}]",
        "p": formatear_p(res.p_value),  
        "¿Significativa?":evaluar_sig(res)
    })
st.subheader("Comparación de regiones")
st.dataframe(pd.DataFrame(filas), hide_index=True)