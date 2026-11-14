"""Convert Giovanni time series (Time Series, Area-Averaged) to the project data contract.

AI-use notice: this script was written with the assistance of Claude (Anthropic)
and reviewed by the team.

Usage, from the repository root:
    python pipeline/02_giovanni_to_contract.py

Reads:
    data/giovanni/<variable>_<region>.csv   e.g. precip_sonora.csv, soil_south_amazon.csv
    data/catalog.csv   one row per variable: product, units, conversion, DOI
    data/regions.csv   one row per region: bounding box (west, south, east, north)

Writes to data/processed/:
    monthly.csv      variable, region, date, value, units, source, doi
    annual.csv       same columns + n_months (annual mean weighted by days in each month)
    series_info.csv  provenance of every series (for the README and the submission form)

Notes:
- Giovanni CSV files do NOT include units. Source units are declared in catalog.csv,
  so do not change the units drop-down in Giovanni: conversions are done here.
- A year is kept only if it has at least MIN_MONTHS_PER_YEAR valid months
  (a project rule of thumb, not a standard).
- If something does not match (different product, different bounding box) the
  script stops: it is better to fail than to silently mix products or regions.
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

MIN_MONTHS_PER_YEAR = 10
EXPECTED_FIRST = "2000-01-01"
EXPECTED_LAST = "2024-12-01"
COVERAGE_WARN_RATIO = 0.8  # warn if the effective box is <80 % of the requested one
PLACEHOLDER_DOIS = {"", "TO_VERIFY"}

# Old file prefixes that are still accepted (earlier downloads used Spanish names).
VARIABLE_ALIASES = {"suelo": "soil"}

# Names of the columns written to data/processed/. Change them here if needed.
OUT_COLUMNS = {"date": "date", "value": "value", "units": "units", "source": "source", "doi": "doi"}


class GiovanniError(Exception):
    """A problem with a file or with the configuration."""


def _floats(text: str) -> list[float]:
    return [float(x) for x in text.split(",")]


def parse_giovanni_csv(path: Path) -> tuple[dict, pd.DataFrame]:
    """Return (metadata, monthly series with columns date, value)."""
    lines = path.read_text(encoding="utf-8-sig").splitlines()

    header_idx = next(
        (i for i, line in enumerate(lines) if line.lower().startswith("time,")), None
    )
    if header_idx is None:
        raise GiovanniError(f"{path.name}: could not find the 'time,...' header row")

    meta: dict[str, str] = {}
    for row in csv.reader(lines[:header_idx]):
        if len(row) >= 2 and row[0].endswith(":"):
            meta[row[0][:-1].strip()] = row[1].strip()

    missing = [k for k in ("User Bounding Box", "URL to Reproduce Results") if k not in meta]
    if missing:
        raise GiovanniError(f"{path.name}: missing metadata {missing}")

    match = re.search(r"[&#]data=([^&]+)", meta["URL to Reproduce Results"])
    if not match:
        raise GiovanniError(f"{path.name}: could not read 'data=' from the Giovanni URL")
    meta["data_id"] = match.group(1)

    fill_key = next((k for k in meta if k.startswith("Fill Value")), None)
    fill_value = float(meta[fill_key]) if fill_key else None

    df = pd.read_csv(path, skiprows=header_idx, skipinitialspace=True)
    df.columns = ["date", "value"]
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    if fill_value is not None:
        df.loc[df["value"] == fill_value, "value"] = np.nan
    return meta, df


def load_config(root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    catalog = pd.read_csv(root / "data" / "catalog.csv", dtype=str).fillna("")
    regions = pd.read_csv(root / "data" / "regions.csv")
    return catalog.set_index("variable"), regions.set_index("region")


def convert(df: pd.DataFrame, cat_row: pd.Series) -> pd.DataFrame:
    out = df.copy()
    out["value"] = out["value"] * float(cat_row["scale"]) + float(cat_row["offset"])
    return out


def to_annual(monthly: pd.DataFrame) -> pd.DataFrame:
    """Annual mean weighted by days in each month; needs MIN_MONTHS_PER_YEAR months."""
    valid = monthly.dropna(subset=["value"]).copy()
    valid["year"] = valid["date"].dt.year
    valid["days"] = valid["date"].dt.days_in_month
    rows = []
    for year, g in valid.groupby("year"):
        if len(g) < MIN_MONTHS_PER_YEAR:
            continue
        rows.append(
            {
                "date": pd.Timestamp(year=int(year), month=1, day=1),
                "value": float(np.average(g["value"], weights=g["days"])),
                "n_months": int(len(g)),
            }
        )
    return pd.DataFrame(rows, columns=["date", "value", "n_months"])


def main(root: Path) -> int:
    catalog, regions = load_config(root)
    files = sorted((root / "data" / "giovanni").glob("*.csv"))
    if not files:
        print("No files found in data/giovanni/")
        return 1

    monthly_parts, annual_parts, info_rows = [], [], []
    ids_seen: dict[str, tuple[str, str]] = {}
    warned: set[str] = set()
    errors, warnings = [], []

    for path in files:
        try:
            variable, _, region = path.stem.partition("_")
            variable = VARIABLE_ALIASES.get(variable, variable)
            if variable not in catalog.index:
                raise GiovanniError(f"{path.name}: variable '{variable}' is not in catalog.csv")
            if region not in regions.index:
                raise GiovanniError(f"{path.name}: region '{region}' is not in regions.csv")

            meta, raw = parse_giovanni_csv(path)
            cat = catalog.loc[variable]

            # 1) the requested box must be the one in regions.csv
            asked = _floats(meta["User Bounding Box"])
            expected = [float(regions.loc[region, c]) for c in ("west", "south", "east", "north")]
            if not np.allclose(asked, expected, atol=1e-6):
                raise GiovanniError(
                    f"{path.name}: requested box {asked} differs from regions.csv {expected}"
                )

            # 2) the product must be the expected one, and the same for every region
            data_id = meta["data_id"]
            if cat["giovanni_data_id"] and data_id != cat["giovanni_data_id"]:
                raise GiovanniError(
                    f"{path.name}: product '{data_id}' != expected "
                    f"'{cat['giovanni_data_id']}' (check resolution and version)"
                )
            if variable in ids_seen and ids_seen[variable][0] != data_id:
                raise GiovanniError(
                    f"{path.name}: '{data_id}' differs from '{ids_seen[variable][0]}' "
                    f"used in {ids_seen[variable][1]}"
                )
            ids_seen.setdefault(variable, (data_id, path.name))
            if not cat["giovanni_data_id"] and f"id:{variable}" not in warned:
                warned.add(f"id:{variable}")
                warnings.append(
                    f"{variable}: catalog.csv has no giovanni_data_id; copy this value: {data_id}"
                )
            if cat["doi"] in PLACEHOLDER_DOIS and f"doi:{variable}" not in warned:
                warned.add(f"doi:{variable}")
                warnings.append(f"{variable}: DOI is still a placeholder in catalog.csv")

            # 3) effective box (the grid trims the requested box)
            got = _floats(meta.get("Data Bounding Box", meta["User Bounding Box"]))
            w_ratio = (got[2] - got[0]) / (asked[2] - asked[0])
            h_ratio = (got[3] - got[1]) / (asked[3] - asked[1])
            if min(w_ratio, h_ratio) < COVERAGE_WARN_RATIO:
                warnings.append(
                    f"{path.name}: the effective box {got} covers only "
                    f"{min(w_ratio, h_ratio):.0%} of the requested one {asked}"
                )

            # 4) expected months
            monthly = convert(raw, cat).dropna(subset=["value"]).sort_values("date")
            expected_idx = pd.date_range(EXPECTED_FIRST, EXPECTED_LAST, freq="MS")
            missing = expected_idx.difference(monthly["date"])
            if len(missing):
                warnings.append(
                    f"{path.name}: {len(missing)} months missing (e.g. {missing[0].date()})"
                )

            annual = to_annual(monthly)

            for df, parts in ((monthly, monthly_parts), (annual, annual_parts)):
                d = df.copy()
                d.insert(0, "region", region)
                d.insert(0, "variable", variable)
                d["units"] = cat["units"]
                d["source"] = cat["source"]
                d["doi"] = cat["doi"]
                parts.append(d)

            info_rows.append(
                {
                    "variable": variable, "region": region, "data_id": data_id,
                    "requested_bbox": asked, "effective_bbox": got,
                    "n_months": len(monthly), "n_years": len(annual),
                    "first_month": monthly["date"].min().date(),
                    "last_month": monthly["date"].max().date(),
                    "file": path.name,
                }
            )
        except GiovanniError as exc:
            errors.append(str(exc))

    for w in warnings:
        print("WARNING:", w)
    for e in errors:
        print("ERROR:", e)
    if errors:
        print(f"\nNothing was written: {len(errors)} error(s).")
        return 1

    out = root / "data" / "processed"
    out.mkdir(parents=True, exist_ok=True)
    base = ["variable", "region", "date", "value", "units", "source", "doi"]
    rename = {"date": OUT_COLUMNS["date"], "value": OUT_COLUMNS["value"],
              "units": OUT_COLUMNS["units"], "source": OUT_COLUMNS["source"],
              "doi": OUT_COLUMNS["doi"]}
    pd.concat(monthly_parts)[base].rename(columns=rename).to_csv(out / "monthly.csv", index=False)
    pd.concat(annual_parts)[base + ["n_months"]].rename(columns=rename).to_csv(out / "annual.csv", index=False)
    pd.DataFrame(info_rows).to_csv(out / "series_info.csv", index=False)
    print(f"OK: {len(files)} series -> {out}")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--root", type=Path, default=Path.cwd(), help="repo root (default: current folder)")
    sys.exit(main(parser.parse_args().root))