"""
AI Generated
Validate and normalize NASA Giovanni monthly area-averaged CSV exports.

Run from repository root:
    python -m pipeline.ingest_giovanni

Original CSV files and existing NSIDC/GISTEMP observations are never changed.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_DIR = PROJECT_ROOT / "data" / "giovanni"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_CSV = OUTPUT_DIR / "giovanni_observations.csv"
CATALOG_JSON = OUTPUT_DIR / "giovanni_catalog.json"

CONTRACT_COLUMNS = ["variable", "region", "date", "value", "units", "source", "doi"]

REGIONS = {
    "chiapas": (-93.5, 16.0, -91.0, 17.5),
    "morelos": (-100.0, 18.0, -98.2, 19.6),
    "suramazonias": (-62.0, -14.0, -52.0, -6.0),
    "sonora": (-111.5, 29.0, -108.8, 31.5),
    "zacatecas": (-103.5, 22.3, -101.5, 24.5),
    "sahel": (-10.0, 12.0, 10.0, 16.0),
}


@dataclass(frozen=True)
class DatasetSpec:
    variable: str
    product: str
    units: str
    source: str
    nominal_resolution_degrees: tuple[float, float]  # latitude, longitude


DATASETS = {
    "precip": DatasetSpec(
        variable="precipitation_rate",
        product="GPM_3IMERGM_07_precipitation",
        units="mm/day",
        source="NASA GPM IMERG Final Run V07 (Giovanni)",
        nominal_resolution_degrees=(0.1, 0.1),
    ),
    "temp": DatasetSpec(
        variable="air_temperature_2m",
        product="M2TMNXSLV_5_12_4_T2M",
        units="°C",
        source="NASA MERRA-2 monthly T2M (Giovanni)",
        nominal_resolution_degrees=(0.5, 0.625),
    ),
    "suelo": DatasetSpec(
        variable="soil_moisture_0_10cm",
        product="GLDAS_NOAH025_M_2_1_SoilMoi0_10cm_inst",
        units="kg/m²",
        source="NASA GLDAS Noah 0–10 cm soil moisture (Giovanni)",
        nominal_resolution_degrees=(0.25, 0.25),
    ),
}

FILE_PATTERN = re.compile(r"^(precip|temp|suelo)_([a-z]+)\.csv$")


class GiovanniValidationError(ValueError):
    """Raised when a Giovanni export fails an explicit integrity check."""


def parse_bbox(raw_bbox: str) -> tuple[float, float, float, float]:
    try:
        bbox = tuple(float(item.strip()) for item in raw_bbox.split(","))
    except ValueError as exc:
        raise GiovanniValidationError(f"Invalid bounding box: {raw_bbox}") from exc
    if len(bbox) != 4:
        raise GiovanniValidationError(f"Expected 4 bounding-box coordinates: {raw_bbox}")
    west, south, east, north = bbox
    if not (-180 <= west < east <= 180 and -90 <= south < north <= 90):
        raise GiovanniValidationError(f"Out-of-range bounding box: {raw_bbox}")
    return bbox  # type: ignore[return-value]


def same_bbox(first: tuple[float, ...], second: tuple[float, ...]) -> bool:
    return all(math.isclose(a, b, abs_tol=1e-6) for a, b in zip(first, second))


def read_export(path: Path) -> tuple[dict[str, str], list[str], list[list[str]]]:
    """Read comma-separated metadata followed by the time-series header and rows."""
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        rows = list(csv.reader(file, skipinitialspace=True))

    metadata: dict[str, str] = {}
    header_index: int | None = None
    for index, row in enumerate(rows):
        if not row or not any(cell.strip() for cell in row):
            continue
        if row[0].strip().lower() == "time":
            header_index = index
            break
        if len(row) >= 2:
            metadata[row[0].strip().rstrip(":")] = ",".join(row[1:]).strip()

    if header_index is None:
        raise GiovanniValidationError(f"Missing time-series header: {path}")
    header = [cell.strip() for cell in rows[header_index]]
    if len(header) != 2 or header[0].lower() != "time":
        raise GiovanniValidationError(f"Expected exactly two columns: {path}")
    data_rows = [row for row in rows[header_index + 1:] if any(cell.strip() for cell in row)]
    if not data_rows:
        raise GiovanniValidationError(f"No observations found: {path}")
    if any(len(row) != 2 for row in data_rows):
        raise GiovanniValidationError(f"Malformed observation row: {path}")
    return metadata, header, data_rows


def parse_metadata_date(value: str, label: str) -> pd.Timestamp:
    try:
        date = pd.to_datetime(value, errors="raise", utc=True)
    except (TypeError, ValueError) as exc:
        raise GiovanniValidationError(f"Invalid {label}: {value}") from exc
    return date.tz_localize(None)

PRODUCT_DOIS = {
    "GPM_3IMERGM_07_precipitation": "10.5067/GPM/IMERG/3B-MONTH/07",
    "M2TMNXSLV_5_12_4_T2M": "10.5067/AP1B0BA5PD2K",
    "GLDAS_NOAH025_M_2_1_SoilMoi0_10cm_inst": "10.5067/SXAVCZFAQLNO",
}


def read_giovanni_csv(path: Path) -> tuple[pd.DataFrame, dict]:
    match = FILE_PATTERN.fullmatch(path.name)
    if match is None:
        raise GiovanniValidationError(f"Unexpected filename: {path.name}")
    dataset_key, region = match.groups()
    if region not in REGIONS:
        raise GiovanniValidationError(f"Unknown region: {region}")
    spec = DATASETS[dataset_key]
    metadata, header, rows = read_export(path)

    required_metadata = {
        "Title", "User Start Date", "User End Date", "User Bounding Box",
        "Data Bounding Box", "URL to Reproduce Results",
    }
    if not required_metadata.issubset(metadata):
        raise GiovanniValidationError(
            f"Missing metadata in {path.name}: {sorted(required_metadata - metadata.keys())}"
        )

    # Allow Giovanni to prefix the product column with "mean_".
    observed_product = header[1].removeprefix("mean_")
    if observed_product != spec.product:
        raise GiovanniValidationError(
            f"Unexpected product in {path.name}: {observed_product} (expected {spec.product})"
        )
    fill_entries = {
        key: value for key, value in metadata.items() if key.startswith("Fill Value (")
    }
    if len(fill_entries) != 1:
        raise GiovanniValidationError(f"Expected one Fill Value entry: {path.name}")
    fill_key, fill_text = next(iter(fill_entries.items()))
    fill_product = fill_key.removeprefix("Fill Value (").removesuffix(")").removeprefix("mean_")
    if fill_product != spec.product:
        raise GiovanniValidationError(f"Fill Value metadata does not match product: {path.name}")
    try:
        fill_value = float(fill_text)
    except ValueError as exc:
        raise GiovanniValidationError(f"Invalid Fill Value: {path.name}") from exc

    requested_bbox = parse_bbox(metadata["User Bounding Box"])
    data_bbox = parse_bbox(metadata["Data Bounding Box"])
    if not same_bbox(requested_bbox, REGIONS[region]):
        raise GiovanniValidationError(f"Unexpected region bounding box: {path.name}")
    if not (
        requested_bbox[0] - 1e-6 <= data_bbox[0] < data_bbox[2] <= requested_bbox[2] + 1e-6
        and requested_bbox[1] - 1e-6 <= data_bbox[1] < data_bbox[3] <= requested_bbox[3] + 1e-6
    ):
        raise GiovanniValidationError(f"Data bounding box outside request: {path.name}")

    reproduction_url = metadata["URL to Reproduce Results"]
    if urlsplit(reproduction_url).hostname != "giovanni.gsfc.nasa.gov":
        raise GiovanniValidationError(f"Unexpected reproduction URL: {path.name}")
    # Giovanni stores query parameters in the URL fragment after '#'.
    query_string = urlsplit(reproduction_url).fragment
    parameters = parse_qs(query_string)
    requested_products = " ".join(parameters.get("data", []))
    if spec.product not in requested_products:
        raise GiovanniValidationError(f"Reproduction URL lacks expected product: {path.name}")
    if "bbox" in parameters and not same_bbox(parse_bbox(parameters["bbox"][0]), requested_bbox):
        raise GiovanniValidationError(f"Reproduction URL bbox mismatch: {path.name}")

    start_date = parse_metadata_date(metadata["User Start Date"], "User Start Date")
    end_date = parse_metadata_date(metadata["User End Date"], "User End Date")
    if start_date > end_date:
        raise GiovanniValidationError(f"Reversed date range: {path.name}")

    dates: list[pd.Timestamp] = []
    values: list[float] = []
    missing_count = 0
    for raw_date, raw_value in rows:
        date = parse_metadata_date(raw_date.strip(), "observation date")
        if date.day != 1 or date.hour != 0 or date.minute != 0 or date.second != 0:
            raise GiovanniValidationError(f"Expected first day of month: {path.name}: {raw_date}")
        if not (start_date.to_period("M") <= date.to_period("M") <= end_date.to_period("M")):
            raise GiovanniValidationError(f"Date outside metadata range: {path.name}: {raw_date}")
        try:
            value = float(raw_value.strip())
        except ValueError as exc:
            raise GiovanniValidationError(f"Invalid numeric value: {path.name}: {raw_value}") from exc
        if not math.isfinite(value) or math.isclose(value, fill_value, abs_tol=1e-8, rel_tol=0):
            missing_count += 1
            value = math.nan
        if dataset_key in ("precip", "suelo") and math.isfinite(value) and value < 0:
            raise GiovanniValidationError(f"Negative physical value: {path.name}: {value}")
        dates.append(date)
        values.append(value)

    if len(dates) != len(set(dates)):
        raise GiovanniValidationError(f"Duplicate monthly dates: {path.name}")
    full_months = pd.period_range(start_date.to_period("M"), end_date.to_period("M"), freq="M")
    available_months = {date.to_period("M") for date in dates}
    missing_months = [str(month) for month in full_months if month not in available_months]


    observations = pd.DataFrame({
        "variable": spec.variable,
        "region": region,
        "date": dates,
        "value": values,
        "units": spec.units,
        "source": spec.source,
        "doi": PRODUCT_DOIS[spec.product],
    })[CONTRACT_COLUMNS]
    # Missing values remain absent: never interpolate or manufacture observations.

    observations = (
        observations
        .dropna(subset=["value"])
        .sort_values("date")
        .reset_index(drop=True)
    )
    if observations.empty:
        raise GiovanniValidationError(f"No valid observations: {path.name}")

    catalog_entry = {
        "file": path.name,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "variable": spec.variable,
        "region": region,
        "product": spec.product,
        "units": spec.units,
        "source": spec.source,
        "nominal_resolution_degrees_lat_lon": list(spec.nominal_resolution_degrees),
        "resolution_note": "Nominal product grid; not an inferred resolution from area-averaged CSV",
        "user_bbox_wsen": list(requested_bbox),
        "data_bbox_wsen": list(data_bbox),
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "raw_rows": len(rows),
        "valid_rows": len(observations),
        "fill_or_nonfinite_count": missing_count,
        "missing_months": missing_months,
        "reproduction_url": reproduction_url,
        "original_metadata": metadata,
    }
    return observations, catalog_entry


def build_giovanni_dataset(input_dir: Path = INPUT_DIR) -> tuple[pd.DataFrame, list[dict]]:
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Giovanni input directory does not exist: {input_dir}")
    paths = sorted(input_dir.glob("*.csv"))
    if not paths:
        raise GiovanniValidationError(f"No Giovanni CSV files in {input_dir}")

    frames: list[pd.DataFrame] = []
    catalog: list[dict] = []
    for path in paths:
        frame, entry = read_giovanni_csv(path)
        frames.append(frame)
        catalog.append(entry)

    combined = pd.concat(frames, ignore_index=True)
    keys = ["variable", "region", "date"]
    if combined.duplicated(keys).any():
        raise GiovanniValidationError("Duplicate observations across Giovanni files")
    combined = combined.sort_values(keys).reset_index(drop=True)
    return combined, catalog


def main() -> None:
    observations, catalog = build_giovanni_dataset()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # Stage outputs before replacement; source files and baseline CSV are untouched.
    staged_csv = OUTPUT_CSV.with_suffix(".csv.tmp")
    staged_catalog = CATALOG_JSON.with_suffix(".json.tmp")
    try:
        observations.to_csv(staged_csv, index=False, date_format="%Y-%m-%d", encoding="utf-8")
        staged_catalog.write_text(json.dumps(catalog, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        staged_csv.replace(OUTPUT_CSV)
        staged_catalog.replace(CATALOG_JSON)
    finally:
        staged_csv.unlink(missing_ok=True)
        staged_catalog.unlink(missing_ok=True)

    print(f"Processed {len(catalog)} Giovanni series; {len(observations)} valid observations")
    print(f"Observations: {OUTPUT_CSV}")
    print(f"Catalog: {CATALOG_JSON}")
    for entry in catalog:
        if entry["fill_or_nonfinite_count"] or entry["missing_months"]:
            print(f"WARNING {entry['file']}: {entry['fill_or_nonfinite_count']} missing values; "
                  f"{len(entry['missing_months'])} missing months")


if __name__ == "__main__":
    main()
