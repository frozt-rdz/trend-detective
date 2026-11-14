"""Write data/processed/series_info.csv: the provenance of every Giovanni series.

Run from the repository root, after the ingest step:
    python -m pipeline.ingest_giovanni
    python -m pipeline.make_series_info

Reads : data/processed/giovanni_catalog.json  (written by pipeline.ingest_giovanni)
Writes: data/processed/series_info.csv        (one row per variable x region)

The file feeds the app ("Data provenance" table) and the submission form
(NASA data used, DOIs, boxes, resolution). It adds nothing that is not in the catalog,
except the derived box size and the approximate number of grid cells in the box.

AI-use notice: written with the assistance of Claude (Anthropic) and reviewed by the team.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CATALOG = ROOT / "data" / "processed" / "giovanni_catalog.json"
OUTPUT = ROOT / "data" / "processed" / "series_info.csv"

# Whether the variable is measured by satellite (with gauges) or comes from a model.
NATURE = {
    "precipitation_rate": "Satellite estimate calibrated with rain gauges",
    "air_temperature_2m": "Reanalysis model (assimilates satellite and ground data)",
    "soil_moisture_0_10cm": "Land-surface model driven by satellite and ground data",
}


def clean_url(url: str) -> str:
    """Drop tracking parameters from the Giovanni reproduction URL."""
    return re.sub(r"\?utm_source=[^#]*", "", url)


def grid_cells(data_bbox: list[float], resolution: list[float]) -> int:
    """Approximate number of grid cells inside the data box (cell-centre extent)."""
    west, south, east, north = data_bbox
    res_lat, res_lon = resolution
    return (round((east - west) / res_lon) + 1) * (round((north - south) / res_lat) + 1)


def build_series_info(catalog: list[dict]) -> pd.DataFrame:
    rows = []
    for item in catalog:
        west, south, east, north = item["user_bbox_wsen"]
        url = item["reproduction_url"]
        data_ids = re.search(r"[#&]data=([^&]*)", url)
        n_datasets = len(re.split(r"%2C|,(?=[A-Za-z])", data_ids.group(1))) if data_ids else 0
        rows.append({
            "variable": item["variable"],
            "region": item["region"],
            "nature": NATURE.get(item["variable"], ""),
            "source": item["source"],
            "product": item["product"],
            "doi": item["doi"],
            "units": item["units"],
            "grid_resolution_deg_lat_lon": "x".join(map(str, item["nominal_resolution_degrees_lat_lon"])),
            "box_west": west, "box_south": south, "box_east": east, "box_north": north,
            "box_width_deg": round(east - west, 2),
            "box_height_deg": round(north - south, 2),
            "approx_grid_cells": grid_cells(item["data_bbox_wsen"], item["nominal_resolution_degrees_lat_lon"]),
            "start_date": item["start_date"][:10],
            "end_date": item["end_date"][:10],
            "valid_months": item["valid_rows"],
            "missing_months": len(item["missing_months"]),
            "file": item["file"],
            "sha256": item["sha256"],
            "giovanni_url": clean_url(url),
            # True when the export URL lists more than one dataset; the CSV itself holds one.
            "url_lists_several_datasets": n_datasets > 1,
        })
    return pd.DataFrame(rows).sort_values(["variable", "region"]).reset_index(drop=True)


def main() -> None:
    catalog = json.loads(CATALOG.read_text(encoding="utf-8"))
    info = build_series_info(catalog)
    info.to_csv(OUTPUT, index=False)
    print(f"Wrote {OUTPUT} ({len(info)} series)")
    flagged = info[info.url_lists_several_datasets]
    if len(flagged):
        print(f"NOTE: {len(flagged)} Giovanni URLs list several datasets (see column "
              "url_lists_several_datasets); each CSV still holds one variable.")


if __name__ == "__main__":
    main()
