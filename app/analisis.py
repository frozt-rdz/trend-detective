from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from stats.trends import trend_report

def formatear_p(p):
    if(p < 0.001):
        return "p < 0.001"
    else:
        return f"p = {p:.3f}"

def analizar_tendencia(df, inicio, fin, ref_inicio = 2000, ref_final = 2019):
    """Tendencia de Sen sobre promedios anuales de anomalías.

    df debe ser la serie COMPLETA de una variable y región (no filtrada por el slider):
    el clima de referencia necesita los años ref_inicio–ref_final. inicio/fin solo
    recortan los años que entran a trend_report. Solo se usan años con 12 meses.
    """
    d = df.dropna(subset=["valor"]).copy()
    # 1. agrega las columnas anio y mes
    d['anio'] = d['fecha'].dt.year
    d['mes'] = d['fecha'].dt.month

    # 2. clima: promedio por mes, SOLO con los años de referencia (groupby)
    ref = d[d['anio'].between(ref_inicio, ref_final)]
    clima = ref.groupby('mes')['valor'].mean()

    # 3. anomalía mensual: valor menos clima (map)
    d['anomalia'] = d['valor'] - d['mes'].map(clima)
    
    # 4. promedio anual de anomalías (groupby) descartando años incompletos (p. ej. el año en curso)
    anual = d.groupby("anio")["anomalia"].agg(["mean", "count"])
    anual = anual[anual["count"] == 12]["mean"]

    # 5. quédate con los años entre inicio y fin
    anual = anual.loc[inicio:fin]

    # 6. regresa DOS cosas: el resultado de trend_report y la serie anual
    resultado = trend_report(anual.index, anual.values)
    return resultado, anual