# AI Generated
"""
NASA Giovanni Data Ingestion Pipeline.

Validates and normalizes NASA Giovanni monthly area-averaged
CSV exports into the Trend Detective data contract.

Run from the repository root:
    python -m pipeline.ingest_giovanni

Outputs:
    data/processed/giovanni_observations.csv
    data/processed/giovanni_catalog.json

Original NASA datasets and existing NSIDC/GISTEMP
observations are never modified.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re

from dataclasses import dataclass
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import pandas as pd


# ============================================================
# PROJECT CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_DIR = PROJECT_ROOT / "data" / "giovanni"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"

OUTPUT_CSV = OUTPUT_DIR / "giovanni_observations.csv"
CATALOG_JSON = OUTPUT_DIR / "giovanni_catalog.json"

CONTRACT_COLUMNS = [
    "variable",
    "region",
    "date",
    "value",
    "units",
    "source",
    "doi",
]


# ============================================================
# REGIONS
# ============================================================

# Bounding boxes use:
# (west, south, east, north)

REGIONS = {
    "chiapas": (-93.5, 16.0, -91.0, 17.5),
    "morelos": (-100.0, 18.0, -98.2, 19.6),
    "south_amazon": (-62.0, -14.0, -52.0, -6.0),
    "sonora": (-111.5, 29.0, -108.8, 31.5),
    "zacatecas": (-103.5, 22.3, -101.5, 24.5),
    "sahel": (-10.0, 12.0, 10.0, 16.0),
}

# Compatibility with older file names.
REGION_ALIASES = {
    "suramazonias": "south_amazon",
}


# ============================================================
# DATASET CONFIGURATION
# ============================================================

@dataclass(frozen=True)
class DatasetSpec:
    variable: str
    product: str
    units: str
    source: str
    nominal_resolution_degrees: tuple[float, float]


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

    "soil": DatasetSpec(
        variable="soil_moisture_0_10cm",
        product="GLDAS_NOAH025_M_2_1_SoilMoi0_10cm_inst",
        units="kg/m²",
        source="NASA GLDAS Noah 0–10 cm soil moisture (Giovanni)",
        nominal_resolution_degrees=(0.25, 0.25),
    ),
}


# ============================================================
# OFFICIAL NASA DATASET DOIS
# ============================================================

PRODUCT_DOIS = {
    "GPM_3IMERGM_07_precipitation":
        "10.5067/GPM/IMERG/3B-MONTH/07",

    "M2TMNXSLV_5_12_4_T2M":
        "10.5067/AP1B0BA5PD2K",

    "GLDAS_NOAH025_M_2_1_SoilMoi0_10cm_inst":
        "10.5067/SXAVCZFAQLNO",
}


# ============================================================
# FILE VALIDATION
# ============================================================

# Accepts:
# precip_chiapas.csv
# temp_morelos.csv
# soil_south_amazon.csv
# suelo_chiapas.csv (legacy)

FILE_PATTERN = re.compile(
    r"^(precip|temp|soil|suelo)_([a-z]+(?:_[a-z]+)*)\.csv$"
)


class GiovanniValidationError(ValueError):
    """Raised when a Giovanni dataset fails validation."""


def parse_bbox(
    raw_bbox: str,
) -> tuple[float, float, float, float]:
    """Parse and validate a geographical bounding box."""

    try:
        bbox = tuple(
            float(item.strip())
            for item in raw_bbox.split(",")
        )
    except ValueError as exc:
        raise GiovanniValidationError(
            f"Invalid bounding box: {raw_bbox}"
        ) from exc

    if len(bbox) != 4:
        raise GiovanniValidationError(
            f"Expected four bounding-box coordinates: {raw_bbox}"
        )

    west, south, east, north = bbox

    if not (
        -180 <= west < east <= 180
        and -90 <= south < north <= 90
    ):
        raise GiovanniValidationError(
            f"Out-of-range bounding box: {raw_bbox}"
        )

    return west, south, east, north


def same_bbox(
    first: tuple[float, ...],
    second: tuple[float, ...],
) -> bool:
    """Compare bounding boxes with a numerical tolerance."""

    return all(
        math.isclose(a, b, abs_tol=1e-6)
        for a, b in zip(first, second)
    )


# ============================================================
# GIOVANNI CSV READER
# ============================================================

def read_export(
    path: Path,
) -> tuple[dict[str, str], list[str], list[list[str]]]:
    """Read Giovanni metadata, header, and observations."""

    with path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        rows = list(
            csv.reader(file, skipinitialspace=True)
        )

    metadata: dict[str, str] = {}
    header_index: int | None = None

    for index, row in enumerate(rows):

        if not row or not any(
            cell.strip() for cell in row
        ):
            continue

        if row[0].strip().lower() == "time":
            header_index = index
            break

        if len(row) >= 2:
            metadata[
                row[0].strip().rstrip(":")
            ] = ",".join(row[1:]).strip()

    if header_index is None:
        raise GiovanniValidationError(
            f"Missing time-series header: {path.name}"
        )

    header = [
        cell.strip()
        for cell in rows[header_index]
    ]

    if len(header) != 2 or header[0].lower() != "time":
        raise GiovanniValidationError(
            f"Expected exactly two observation columns: {path.name}"
        )

    data_rows = [
        row
        for row in rows[header_index + 1:]
        if any(cell.strip() for cell in row)
    ]

    if not data_rows:
        raise GiovanniValidationError(
            f"No observations found: {path.name}"
        )

    if any(len(row) != 2 for row in data_rows):
        raise GiovanniValidationError(
            f"Malformed observation row: {path.name}"
        )

    return metadata, header, data_rows


def parse_metadata_date(
    value: str,
    label: str,
) -> pd.Timestamp:
    """Parse an ISO date into a timezone-naive UTC timestamp."""

    try:
        date = pd.to_datetime(
            value,
            errors="raise",
            utc=True,
        )
    except (TypeError, ValueError) as exc:
        raise GiovanniValidationError(
            f"Invalid {label}: {value}"
        ) from exc

    if pd.isna(date):
        raise GiovanniValidationError(f"Missing {label}")
    return date.tz_localize(None)


# ============================================================
# DATASET NORMALIZATION
# ============================================================

def read_giovanni_csv(
    path: Path,
) -> tuple[pd.DataFrame, dict]:
    """Validate one Giovanni CSV and normalize its observations."""

    match = FILE_PATTERN.fullmatch(path.name)

    if match is None:
        raise GiovanniValidationError(
            f"Unexpected filename: {path.name}"
        )

    dataset_key, raw_region = match.groups()

    # Normalize legacy dataset names.
    if dataset_key == "suelo":
        dataset_key = "soil"

    # Normalize legacy region names.
    region = REGION_ALIASES.get(
        raw_region,
        raw_region,
    )

    if region not in REGIONS:
        raise GiovanniValidationError(
            f"Unknown region: {region}"
        )

    spec = DATASETS[dataset_key]

    metadata, header, rows = read_export(path)

    # --------------------------------------------------------
    # Metadata validation
    # --------------------------------------------------------

    required_metadata = {
        "Title",
        "User Start Date",
        "User End Date",
        "User Bounding Box",
        "Data Bounding Box",
        "URL to Reproduce Results",
    }

    missing_metadata = (
        required_metadata - metadata.keys()
    )

    if missing_metadata:
        raise GiovanniValidationError(
            f"Missing metadata in {path.name}: "
            f"{sorted(missing_metadata)}"
        )

    # --------------------------------------------------------
    # Product validation
    # --------------------------------------------------------

    observed_product = header[1].removeprefix(
        "mean_"
    )

    if observed_product != spec.product:
        raise GiovanniValidationError(
            f"Unexpected product in {path.name}: "
            f"{observed_product}. "
            f"Expected: {spec.product}"
        )

    # --------------------------------------------------------
    # Fill value validation
    # --------------------------------------------------------

    fill_entries = {
        key: value
        for key, value in metadata.items()
        if key.startswith("Fill Value (")
    }

    if len(fill_entries) != 1:
        raise GiovanniValidationError(
            f"Expected one Fill Value entry: {path.name}"
        )

    fill_key, fill_text = next(
        iter(fill_entries.items())
    )

    fill_product = (
        fill_key
        .removeprefix("Fill Value (")
        .removesuffix(")")
        .removeprefix("mean_")
    )

    if fill_product != spec.product:
        raise GiovanniValidationError(
            f"Fill Value product mismatch: {path.name}"
        )

    try:
        fill_value = float(fill_text)
    except ValueError as exc:
        raise GiovanniValidationError(
            f"Invalid Fill Value: {path.name}"
        ) from exc

    # --------------------------------------------------------
    # Bounding box validation
    # --------------------------------------------------------

    if not math.isfinite(fill_value) or not metadata["Title"].strip():
        raise GiovanniValidationError(f"Invalid title or fill value: {path.name}")

    requested_bbox = parse_bbox(
        metadata["User Bounding Box"]
    )

    data_bbox = parse_bbox(
        metadata["Data Bounding Box"]
    )

    if not same_bbox(
        requested_bbox,
        REGIONS[region],
    ):
        raise GiovanniValidationError(
            f"Unexpected region bounding box: {path.name}"
        )

    west, south, east, north = requested_bbox

    data_west, data_south, data_east, data_north = data_bbox

    tolerance = 1e-6

    if not (
        west - tolerance <= data_west
        < data_east <= east + tolerance
        and south - tolerance <= data_south
        < data_north <= north + tolerance
    ):
        raise GiovanniValidationError(
            f"Data bounding box outside request: {path.name}"
        )

    # --------------------------------------------------------
    # Reproduction URL validation
    # --------------------------------------------------------

    reproduction_url = metadata[
        "URL to Reproduce Results"
    ]

    parsed_url = urlsplit(reproduction_url)

    if (parsed_url.scheme != "https" or parsed_url.hostname != "giovanni.gsfc.nasa.gov"
            or parsed_url.path.rstrip("/") != "/giovanni"):
        raise GiovanniValidationError(
            f"Unexpected reproduction URL: {path.name}"
        )

    parameters = parse_qs(
        parsed_url.fragment
    )

    expected_query = {
        "precip": spec.product + "(units=mm/day)",
        "temp": spec.product + "(units=C)",
        "soil": spec.product,
    }[dataset_key]
    if expected_query not in parameters.get("data", [""])[0].split(","):
        raise GiovanniValidationError(f"Reproduction URL product or units mismatch: {path.name}")
    if parameters.get("service") != ["ArAvTs"]:
        raise GiovanniValidationError(f"Unexpected reproduction service: {path.name}")
    url_bbox = parse_bbox(parameters.get("bbox", [""])[0])
    if not same_bbox(url_bbox, requested_bbox):
        raise GiovanniValidationError(f"Reproduction URL bbox mismatch: {path.name}")

    # --------------------------------------------------------
    # Date range validation
    # --------------------------------------------------------

    start_date = parse_metadata_date(
        metadata["User Start Date"],
        "User Start Date",
    )

    end_date = parse_metadata_date(
        metadata["User End Date"],
        "User End Date",
    )

    for key, expected in (("starttime", start_date), ("endtime", end_date)):
        if parse_metadata_date(parameters.get(key, [""])[0], key) != expected:
            raise GiovanniValidationError(f"Reproduction URL {key} mismatch: {path.name}")

    if start_date > end_date:
        raise GiovanniValidationError(
            f"Reversed date range: {path.name}"
        )

    # --------------------------------------------------------
    # Observation validation
    # --------------------------------------------------------

    dates: list[pd.Timestamp] = []
    values: list[float] = []

    missing_count = 0

    for raw_date, raw_value in rows:

        date = parse_metadata_date(
            raw_date.strip(),
            "observation date",
        )

        if (
            date.day != 1
            or date.hour != 0
            or date.minute != 0
            or date.second != 0
            or date.microsecond != 0
            or date.nanosecond != 0
        ):
            raise GiovanniValidationError(
                f"Expected first day of month: "
                f"{path.name}: {raw_date}"
            )

        if not (
            start_date.to_period("M")
            <= date.to_period("M")
            <= end_date.to_period("M")
        ):
            raise GiovanniValidationError(
                f"Date outside metadata range: "
                f"{path.name}: {raw_date}"
            )

        try:
            value = float(
                raw_value.strip()
            )
        except ValueError as exc:
            raise GiovanniValidationError(
                f"Invalid numeric value: "
                f"{path.name}: {raw_value}"
            ) from exc

        # Preserve missingness rather than inventing observations.
        if (
            not math.isfinite(value)
            or math.isclose(
                value,
                fill_value,
                abs_tol=1e-8,
                rel_tol=0,
            )
        ):
            missing_count += 1
            value = math.nan

        # Negative precipitation rates or soil water mass
        # are not physically valid for these products.
        if (
            dataset_key in ("precip", "soil")
            and math.isfinite(value)
            and value < 0
        ):
            raise GiovanniValidationError(
                f"Negative physical value: "
                f"{path.name}: {value}"
            )

        dates.append(date)
        values.append(value)

    # --------------------------------------------------------
    # Duplicate and missing month validation
    # --------------------------------------------------------

    if len(dates) != len(set(dates)):
        raise GiovanniValidationError(
            f"Duplicate monthly dates: {path.name}"
        )

    full_months = pd.period_range(
        start_date.to_period("M"),
        end_date.to_period("M"),
        freq="M",
    )

    available_months = {
        date.to_period("M")
        for date in dates
    }

    missing_months = [
        str(month)
        for month in full_months
        if month not in available_months
    ]

    # --------------------------------------------------------
    # Normalize observations
    # --------------------------------------------------------

    observations = pd.DataFrame({
        "variable": spec.variable,
        "region": region,
        "date": dates,
        "value": values,
        "units": spec.units,
        "source": spec.source,
        "doi": PRODUCT_DOIS[spec.product],
    })[CONTRACT_COLUMNS]

    observations = (
        observations
        .dropna(subset=["value"])
        .sort_values("date")
        .reset_index(drop=True)
    )

    if observations.empty:
        raise GiovanniValidationError(
            f"No valid observations: {path.name}"
        )

    # --------------------------------------------------------
    # Reproducible metadata catalog
    # --------------------------------------------------------

    catalog_entry = {
        "file": path.name,
        "sha256": hashlib.sha256(
            path.read_bytes()
        ).hexdigest(),
        "variable": spec.variable,
        "region": region,
        "product": spec.product,
        "doi": PRODUCT_DOIS[spec.product],
        "units": spec.units,
        "source": spec.source,
        "nominal_resolution_degrees_lat_lon": list(
            spec.nominal_resolution_degrees
        ),
        "resolution_note": (
            "Nominal product grid; not an inferred "
            "resolution from area-averaged CSV"
        ),
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


# ============================================================
# BUILD COMPLETE DATASET
# ============================================================

def build_giovanni_dataset(
    input_dir: Path = INPUT_DIR,
) -> tuple[pd.DataFrame, list[dict]]:
    """Process all available Giovanni CSV files."""

    if not input_dir.is_dir():
        raise FileNotFoundError(
            f"Giovanni input directory does not exist: {input_dir}"
        )

    paths = sorted(
        input_dir.glob("*.csv")
    )

    if not paths:
        raise GiovanniValidationError(
            f"No Giovanni CSV files in {input_dir}"
        )

    frames: list[pd.DataFrame] = []
    catalog: list[dict] = []

    series_files = {}
    for path in paths:

        print(f"Processing: {path.name}")

        frame, entry = read_giovanni_csv(
            path
        )

        key = entry["variable"], entry["region"]
        if key in series_files:
            raise GiovanniValidationError(
                f"Duplicate canonical series {key}: {series_files[key]} and {path.name}"
            )
        series_files[key] = path.name
        frames.append(frame)
        catalog.append(entry)

    combined = pd.concat(
        frames,
        ignore_index=True,
    )

    keys = [
        "variable",
        "region",
        "date",
    ]

    if combined.duplicated(keys).any():
        raise GiovanniValidationError(
            "Duplicate observations across Giovanni files"
        )

    combined = (
        combined
        .sort_values(keys)
        .reset_index(drop=True)
    )

    return combined, catalog


# ============================================================
# MAIN EXECUTION
# ============================================================

def main() -> None:
    """Build and export validated Giovanni datasets."""

    observations, catalog = build_giovanni_dataset()

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    staged_csv = OUTPUT_CSV.with_suffix(
        ".csv.tmp"
    )

    staged_catalog = CATALOG_JSON.with_suffix(
        ".json.tmp"
    )

    try:
        observations.to_csv(
            staged_csv,
            index=False,
            date_format="%Y-%m-%d",
            encoding="utf-8",
        )

        staged_catalog.write_text(
            json.dumps(
                catalog,
                ensure_ascii=False,
                indent=2,
            ) + "\n",
            encoding="utf-8",
        )

        staged_csv.replace(
            OUTPUT_CSV
        )

        staged_catalog.replace(
            CATALOG_JSON
        )

    finally:
        staged_csv.unlink(
            missing_ok=True
        )

        staged_catalog.unlink(
            missing_ok=True
        )

    print()
    print("=" * 60)
    print("NASA GIOVANNI PIPELINE COMPLETED")
    print("=" * 60)

    print(
        f"Processed series: {len(catalog)}"
    )

    print(
        f"Valid observations: {len(observations)}"
    )

    print(
        f"Observations: {OUTPUT_CSV}"
    )

    print(
        f"Catalog: {CATALOG_JSON}"
    )

    for entry in catalog:

        if (
            entry["fill_or_nonfinite_count"]
            or entry["missing_months"]
        ):
            print(
                f"WARNING {entry['file']}: "
                f"{entry['fill_or_nonfinite_count']} "
                f"missing values; "
                f"{len(entry['missing_months'])} "
                f"missing months"
            )


if __name__ == "__main__":
    main()
