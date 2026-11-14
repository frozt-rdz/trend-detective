# Trend Detective

**Is the Earth changing where you live?** An interactive tool that picks variables measured by
NASA Earth-science missions and models, shows how they change over time, and tests whether the
change is statistically significant.

NASA Space Apps Challenge 2026 · *Be an Earth System Trend Detective*

> Live app: _add the public Streamlit link here_

## What it does

For three variables (precipitation, soil moisture, air temperature) in six regions
(2000–2024, monthly NASA data from Giovanni) the app answers the four questions of the challenge:

| Question | How the app answers it |
|---|---|
| What is changing? | Pick a variable; plain-language definitions are in the app |
| Where? | Pick one of six boxes (four in Mexico, plus Sahel and Southern Amazon as contrast) |
| How much? | Sen's slope per decade with a 95 % interval, and a yearly chart |
| Is it significant? | Mann-Kendall / Hamed-Rao p-value, corrected for 18 comparisons (Benjamini-Hochberg q) |

Regions were chosen by the data, not by labels such as "dry" or "rainy": a very rainy region can be
drying. The app states the size of each box and how many grid cells it covers.

## Main findings (2000–2024)

- **Precipitation moves in opposite directions:** Sahel **+41.7 mm/decade** (q = 0.028) and Southern
  Amazon **−143 mm/decade** (q = 0.001). Both stay significant after correcting for 18 comparisons.
- **The four Mexican regions show no significant precipitation trend** after the correction
  (Chiapas: p = 0.042, q = 0.069). A non-significant result is a result: 25 years is short and
  Mexican rainfall varies with ENSO.
- Soil moisture is falling significantly in Chiapas and Southern Amazon; air temperature is rising
  significantly in Chiapas, Morelos, Zacatecas and Southern Amazon.
- Robustness: from 2005 the Sahel increase is no longer significant (p = 0.18) while the Southern
  Amazon decline remains (p = 0.002). The app shows this check for every series.

## Run it locally

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m pipeline.ingest_giovanni      # data/giovanni/*.csv -> giovanni_observations.csv
python -m pipeline.make_series_info     # provenance of every series -> series_info.csv
python -m stats.analyze_giovanni        # optional: writes giovanni_significance.csv
python -m streamlit run app/main.py
```

## Project structure

| Path | Purpose |
|---|---|
| `data/giovanni/` | The 18 original Giovanni exports (`temp_`, `precip_`, `soil_` + region) |
| `data/catalog.csv`, `data/regions.csv` | Product units/DOIs and region boxes |
| `pipeline/ingest_giovanni.py` | Validates exports and writes the common data contract + `giovanni_catalog.json` |
| `pipeline/make_series_info.py` | Writes `series_info.csv`: source, DOI, box, grid cells, checksum, reproduction URL |
| `stats/trends.py` | `trend_report`: Sen's slope, Mann-Kendall / Hamed-Rao |
| `stats/analyze_giovanni.py` | Annual series, the 18 planned tests and FDR correction |
| `app/main.py` | The Streamlit app |
| `stats/tests/` | Automated tests |

Data contract (`giovanni_observations.csv`): `variable, region, date, value, units, source, doi`.

## Methods

1. **Annual values.** Only calendar years with all 12 valid months are used. Precipitation is the sum
   of monthly mean rates times days in each month (mm/year); temperature and soil moisture are the mean
   of the 12 monthly values.
2. **Slope.** Sen's slope on actual calendar years, reported per decade, with the conventional 95 %
   interval from `scipy.stats.theilslopes`. The interval does not account for autocorrelation.
3. **Significance.** Mann-Kendall by default. If the lag-1 autocorrelation of the residuals is
   **positive and above 0.2**, Hamed-Rao is used instead. Hamed-Rao corrects for *positive* serial
   correlation; with negative correlation its correction can inflate the statistic or make the variance
   negative (for example Sahel precipitation: Mann-Kendall p = 0.012 against an invalid Hamed-Rao
   variance), so negative autocorrelation falls back to the original test. The 0.2 threshold is a
   practical heuristic, not a universal criterion.
4. **Multiple comparisons.** The 18 planned comparisons (3 variables × 6 regions) are corrected
   together with Benjamini-Hochberg (q-value). A result is called significant when q ≤ 0.05.
5. **Robustness.** The app repeats the test from 2005 for the selected series. It is a check, not part
   of the 18 planned tests.

Simulated check (3 000 series of 25 years, no real trend, α = 0.05): false positives with this rule are
4.6 % for white noise and 0.3 % with negative autocorrelation. With strong *positive* autocorrelation
(+0.5) about 18 % of series still pass, which is one reason 25 years is limited evidence.

## Limitations

- 25 annual points per series; interannual cycles such as ENSO can mimic or hide a trend.
- Boxes are averages over many grid cells and trends can differ inside a box.
- Temperature (MERRA-2) and soil moisture (GLDAS) are model outputs fed with observations;
  precipitation (IMERG) is a satellite estimate calibrated with rain gauges.
- The 2-m temperature in the Sahel shows a small decrease (−0.24 °C/decade, not significant after
  correction); it should be checked against another source before being interpreted.
- Linear trends only.
- Some Giovanni reproduction URLs list more than one dataset (flag `url_lists_several_datasets` in
  `series_info.csv`); each CSV still contains a single variable.

## Tests

```powershell
python -m pytest -v
```

They cover the trend statistics (including the negative-autocorrelation rule), the ingestion contract,
the annual aggregation, the FDR correction, the Sahel and Southern Amazon regression values, the
provenance table and a smoke test of the app for every variable and region.

## Data sources

All data come from NASA Giovanni (time series, area-averaged, monthly, 2000–2024):

- Beaudoing, H., Rodell, M., & NASA/GSFC/HSL (2020). GLDAS Noah Land Surface Model L4 monthly 0.25 x 0.25 degree, Version 2.1 [Data set]. NASA GES DISC. https://doi.org/10.5067/SXAVCZFAQLNO
- Huffman, G. J., Stocker, E. F., Bolvin, D. T., Nelkin, E. J., & Tan, J. (2023). GPM IMERG Final Precipitation L3 1 month 0.1 degree x 0.1 degree V07 [Data set]. NASA GES DISC. https://doi.org/10.5067/GPM/IMERG/3B-MONTH/07
- Global Modeling and Assimilation Office & Pawson, S. (2015). MERRA-2 tavgM_2d_slv_Nx: 2d, Monthly mean, Time-averaged, Single-Level, Assimilation, Single-Level Diagnostics V5.12.4 [Data set]. NASA GES DISC. https://doi.org/10.5067/AP1B0BA5PD2K

Software: Streamlit, pandas, NumPy, SciPy, pyMannKendall, Plotly, pytest (open source).
This project uses NASA open data and contains no NASA logos or branding; it is not endorsed by NASA.

## Use of AI

Parts of the code (the trend module, the Giovanni pipeline and analysis, the Streamlit app, the
provenance script and the tests) were written with the assistance of Claude (Anthropic) and reviewed
by the team; files carry an AI-use notice. The project idea, the selection of variables and regions
and the interpretation of results are the team's own.
