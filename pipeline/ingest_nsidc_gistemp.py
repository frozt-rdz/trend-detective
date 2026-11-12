"""
Descargar y normaliza dos series.

Ejecutar DESDE la raiz del repositorio:
    python pipeline/ingest_nsidc_gistemp.py

Salida: data/processed/observaciones.csv
Fuentes: NSIDC G02135 v4.0 y NASA GISS GISTEMP v4.
"""
from __future__ import annotations

from pathlib import Path
from urllib.request import Request, urlopen

import pandas as pd

BASE = Path(__file__).resolve().parents[1]
RAW = BASE / "data" / "raw"
OUTPUT = BASE / "data" / "processed" / "observaciones.csv"
START_YEAR = 2000
END_YEAR = 2026

NSIDC_PREFIX = "https://noaadata.apps.nsidc.org/NOAA/G02135/north/monthly/data"
GISTEMP_URL = "https://data.giss.nasa.gov/gistemp/tabledata_v4/NH.Ts%2BdSST.csv"
NSIDC_DOI = "10.7265/a98x-0f50"
GISTEMP_REFERENCE = "https://data.giss.nasa.gov/gistemp/"
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
CONTRACT = ["variable", "region", "fecha", "valor", "unidades", "fuente", "doi"]


def fetch_csv(url: str, output_path: Path) -> Path:
    """Descarga una fuente oficial o utiliza una copia local previamente descargada."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and output_path.stat().st_size > 0:
        print(f"Usando descarga local: {output_path}")
        return output_path

    print(f"Descargando: {url}")
    request = Request(url, headers={"User-Agent": "Mozilla/5.0 TrendDetectiveResearch/1.0"})
    with urlopen(request, timeout=45) as response:
        content = response.read()
    if not content or content.lstrip().startswith(b"<"):
        raise ValueError(f"La fuente no devolvió un CSV válido: {url}")
    output_path.write_bytes(content)
    return output_path


def parse_nsidc_month(path: Path, month: int) -> pd.DataFrame:
    """Convierte el archivo mensual NSIDC a filas de fecha y extensión de hielo."""
    source = pd.read_csv(path, skipinitialspace=True)
    source.columns = source.columns.str.strip().str.lower()
    required = {"year", "mo", "extent"}
    if not required.issubset(source.columns):
        raise ValueError(f"Columnas NSIDC inesperadas: {source.columns.tolist()}")

    years = pd.to_numeric(source["year"], errors="coerce")
    months = pd.to_numeric(source["mo"], errors="coerce")
    extent = pd.to_numeric(source["extent"], errors="coerce")
    good = (years.between(START_YEAR, END_YEAR) & (months == month) & extent.notna() & (extent >= 0))
    result = pd.DataFrame({
        "fecha": pd.to_datetime({
            "year": years[good].astype(int),
            "month": months[good].astype(int),
            "day": 1,
        }),
        "valor": extent[good].astype(float),
    })
    if result["fecha"].duplicated().any():
        raise ValueError(f"NSIDC contiene meses duplicados: {path}")
    if result.empty:
        raise ValueError(f"NSIDC no devolvió observaciones válidas: {path}")

    result["variable"] = "extension_hielo_marino"
    result["region"] = "Artico"
    result["unidades"] = "millones km²"
    result["fuente"] = "NSIDC Sea Ice Index G02135 v4.0"
    result["doi"] = NSIDC_DOI
    return result[CONTRACT]


def parse_gistemp(path: Path) -> pd.DataFrame:
    """Lee anomalías mensuales respecto a 1951-1980; descarta valores '***'."""
    # La primera línea del archivo GISTEMP es un título descriptivo.
    df = pd.read_csv(path, skiprows=1, dtype=str, skipinitialspace=True)
    df.columns = df.columns.str.strip()
    required = {"Year", *MONTHS}
    if not required.issubset(df.columns):
        raise ValueError(f"Columnas GISTEMP inesperadas: {df.columns.tolist()}")

    df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
    df = df[df["Year"].between(START_YEAR, END_YEAR)].copy()
    long = df.melt(id_vars="Year", value_vars=MONTHS, var_name="mes", value_name="valor")
    long["valor"] = pd.to_numeric(long["valor"], errors="coerce")
    long = long.dropna(subset=["valor"]).copy()
    long["month_number"] = long["mes"].map({name: i for i, name in enumerate(MONTHS, 1)})
    long["fecha"] = pd.to_datetime({
        "year": long["Year"].astype(int),
        "month": long["month_number"].astype(int),
        "day": 1,
    })
    if long["fecha"].duplicated().any():
        raise ValueError("GISTEMP contiene fechas duplicadas")
    if long.empty:
        raise ValueError("GISTEMP no devolvió observaciones válidas")

    long["variable"] = "anomalia_temperatura"
    long["region"] = "Hemisferio norte"
    long["unidades"] = "°C"
    long["fuente"] = "NASA GISS GISTEMP v4; base 1951-1980"
    long["doi"] = "No asignado; ver fuente oficial"
    return long[CONTRACT]


def build_observations() -> pd.DataFrame:
    frames = []
    for month in range(1, 13):
        name = f"N_{month:02d}_extent_v4.0.csv"
        path = fetch_csv(f"{NSIDC_PREFIX}/{name}", RAW / "nsidc" / name)
        frames.append(parse_nsidc_month(path, month))

    gistemp_path = fetch_csv(GISTEMP_URL, RAW / "gistemp" / "NH.Ts+dSST.csv")
    frames.append(parse_gistemp(gistemp_path))
    output = pd.concat(frames, ignore_index=True)
    output = output.sort_values(["variable", "region", "fecha"], ignore_index=True)
    if output.duplicated(subset=["variable", "region", "fecha"]).any():
        raise ValueError("Hay observaciones duplicadas para la misma serie y fecha")
    if output["variable"].nunique() != 2:
        raise ValueError("No se cargaron las dos variables previstas")
    return output


def main() -> None:
    output = build_observations()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(OUTPUT, index=False, date_format="%Y-%m-%d", encoding="utf-8-sig")
    print(f"\nCSV creado: {OUTPUT}")
    for (variable, region), sample in output.groupby(["variable", "region"]):
        print(f"- {variable} / {region}: {len(sample)} meses ({sample['fecha'].min():%Y-%m} a {sample['fecha'].max():%Y-%m})")
    print("IMPORTANTE: los últimos años pueden estar incompletos. Para análisis anual exija 12 meses.")


if __name__ == "__main__":
    main()
