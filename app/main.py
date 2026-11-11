from pathlib import Path
import pandas as pd
import streamlit as st
import numpy as np

st.set_page_config(page_title="Detective de tendencias", layout="wide")

@st.cache_data #streamlit recarga todo el codigo, con esto guarda los resultados del CSV y no los relee con cada clic
def cargar():
    ruta = Path(__file__).parent.parent / "data" / "sample" / "muestra_sintetica.csv" #ruta de los datos
    return pd.read_csv(ruta, parse_dates=["fecha"]) #pasa las fechas a algo digerible

df = cargar() #llamamos a la funcion

col1, col2 = st.columns(2) #añadimos a la app dos columnas

variable = col1.selectbox("Variable", sorted(df['variable'].unique()))
region = col2.selectbox("Region", sorted(df['region'].unique()))

serie = df[(df['region'] == region) & (df['variable'] == variable)].dropna(subset=["valor"]).sort_values("fecha")
#serie de datos con la que se trabajara

unidad = serie["unidades"].iloc[0] #sale de los datos

#años en decimales para que pendiente salga en unidades/AÑO
x = serie["fecha"].dt.year + (serie["fecha"].dt.month - 1) / 12

b, a = np.polyfit(x, serie["valor"], 1)     # ajusta una recta: b = pendiente, a = ordenada
serie["tendencia"] = a + (b*x) #ecuacion de la recta y = a + b·x evaluada en cada x

deci_var = {"temperatura_superficial" : 3, "humedad_suelo": 4} #diccionario para el num de decimales por cada variable
d = deci_var[variable]

st.metric("Tendencia", f"{b * 10:+.{d}f} {unidad}/década")
st.line_chart(serie.set_index("fecha")[["valor", "tendencia"]])

