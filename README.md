# Trend Detective

A tool for detecting trends in Earth system variables.

**NASA Space Apps Challenge 2026: Be An Earth System Trend Detective!**

## How to Run

### 1. Create a Virtual Environment

```powershell
python -m venv .venv
```

### 2. Activate the Virtual Environment

```powershell
.venv\Scripts\Activate.ps1
```

### 3. Install Dependencies

```powershell
pip install -r requirements.txt
```

### 4. Run the Application

```powershell
streamlit run app/main.py
```

---

## Statistical Trend Analysis

The `stats/trends.py` module contains the `trend_report` function, designed to analyze trends in annual time series.

### Methods Used

- **Sen's slope:** estimates the magnitude and direction of the trend using the median of the slopes between pairs of observations.
- **Sen's confidence interval:** estimates the range of plausible slope values using `scipy.stats.theilslopes`.
- **Mann-Kendall:** evaluates whether a monotonic trend exists in the time series.
- **Hamed-Rao:** applies a Mann-Kendall modification to account for serial autocorrelation.

### Usage

```python
from stats.trends import trend_report

years = [2000, 2001, 2002, 2003, 2004, 2005]
values = [10.2, 10.5, 10.8, 11.1, 11.4, 11.7]

result = trend_report(years, values)

print("Tendencia:", result.trend)
print("Pendiente por década:", result.sen_slope_decade)
print("Valor p:", result.p_value)
print("Significativa:", result.significant)
```

The function accepts:

| Parameter | Description |
|---|---|
| `years` | Years corresponding to the observations. |
| `values` | Values of the variable being analyzed. |
| `alpha` | Statistical significance level (default: 0.05). |

The `TrendResult` output contains:

- `n`: number of valid observations.
- `sen_slope_decade`: Sen's slope per decade.
- `ci_low_decade` and `ci_high_decade`: confidence interval bounds.
- `p_value`: p-value of the selected test.
- `z`: Z statistic.
- `trend`: direction of the trend.
- `significant`: indicates whether the result is statistically significant.
- `lag1_autocorr`: lag-1 residual autocorrelation.
- `method`: statistical method used.
- `warnings`: analysis warnings.

### Methodological Considerations

1. Non-finite values (`NaN` and infinities) are removed together with their corresponding years.
2. The slope and confidence interval are expressed in units per decade.
3. Lag-1 autocorrelation is calculated on the residuals of the estimated trend, avoiding confusion between an upward or downward trend and residual autocorrelation.
4. A heuristic threshold of `|r1| > 0.2` is used to decide when to apply Hamed-Rao. This is not a formal test of autocorrelation significance, nor is it a universal threshold.
5. A computational minimum of four observations is required to estimate autocorrelation, although such a small sample does not guarantee a reliable estimate.
6. Observations should represent an appropriate time sequence. Missing years can affect the interpretation of autocorrelation and tests that treat observations as consecutive.

Statistical significance does not necessarily mean that the magnitude of the trend is environmentally important.

---

## Automated Tests

The module's tests are located in `stats/tests/test_trends.py`.

To run them from the project root:

```powershell
python -m pytest -v
```

Test cases include:

- A known increasing trend.
- White noise.
- An AR(1) series with no deterministic trend.
- `alpha` validation.
- Duplicate year detection.
- Removal of `NaN` values.

---

## Data Sources

Beaudoing, H., Rodell, M., & NASA/GSFC/HSL (2020). GLDAS Noah Land Surface Model L4 monthly 0.25 x 0.25 degree, Version 2.1 [Data set]. NASA GES DISC. https://doi.org/10.5067/SXAVCZFAQLNO

Fetterer, F., Knowles, K., Meier, W. N., Savoie, M., Windnagel, A. K., & Stafford, T. (2025). Sea Ice Index (G02135, Version 4) [Dataset]. National Snow and Ice Data Center. https://doi.org/10.7265/a98x-0f50

GISTEMP Team. (2026). GISS Surface Temperature Analysis (GISTEMP), version 4 [Dataset]. NASA Goddard Institute for Space Studies. Accessed November 11, 2026. https://data.giss.nasa.gov/gistemp/

Huffman, G. J., Stocker, E. F., Bolvin, D. T., Nelkin, E. J., & Tan, J. (2023). GPM IMERG Final Precipitation L3 1 month 0.1 degree x 0.1 degree V07 [Data set]. NASA Goddard Earth Sciences Data and Information Services Center. https://doi.org/10.5067/GPM/IMERG/3B-MONTH/07 

Global Modeling and Assimilation Office & Pawson, S. (2015). MERRA-2 tavgM_2d_slv_Nx ... V5.12.4 [Data set]. NASA GES DISC. https://doi.org/10.5067/AP1B0BA5PD2K

Lenssen, N., Schmidt, G. A., Hendrickson, M., Jacobs, P., Menne, M. J., & Ruedy, R. (2024). A NASA GISTEMPv4 observational uncertainty ensemble. Journal of Geophysical Research: Atmospheres, 129, e2023JD040179. https://doi.org/10.1029/2023JD040179
