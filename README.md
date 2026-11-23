# DS4420_Proj1: Pollen monitoring

Pollen monitoring has two parts: identifying which pollen is in the air and
predicting how much there will be. This repo covers the second part:
forecasting daily **tree pollen in Louisville, KY** from the previous day's
count, the time of year and the weather. The pollen-identification CNN is
being built separately.

## Repository layout

```
data/
  weather_data.csv              NOAA daily weather, Louisville airport (USW00093821), 2003-2022
  louisville_pollen_daily.csv   CDC/NAB daily tree and weed pollen counts (grains/m³), 2003-2022
  louisville_humidity_daily.csv Daily relative humidity and dew point (Open-Meteo, ERA5 reanalysis)
  louisville_merged.csv         All of the above joined by date, plus engineered features
scripts/
  fetch_humidity.py             Downloads the humidity file
  build_dataset.py              Builds louisville_merged.csv
  bayes_regression.py           Method 1: Bayesian linear regression (NumPy only)
  mlp.py                        Method 2: multilayer perceptron (NumPy only)
  make_figures.py               Report/poster figures (fig_*.png)
figures/
  fig_season.png                The average tree pollen season (report/poster)
  fig_forecast_2019.png         2019 forecast with 90% intervals (report/poster)
  fig_drivers.png               What drives the forecast (report/poster)
  fig_accuracy.png              Both models vs baselines, R² (report/poster)
  forecast_2019.png             Bayesian forecast with 90% intervals, 2019 season
  coefficients.png              Posterior effect of each input
  mlp_vs_bayes_2019.png         Both models' forecasts, 2019 season
  mlp_training_curve.png        MLP train/validation error by epoch
```

## Setup and running

Tested with Python 3.12. From the repo root:

```bash
python -m venv .venv
.venv/bin/pip install -r requirements.txt
```

The data files are already committed, so you can go straight to the models.
Each script prints every metric quoted below and in the report:

```bash
.venv/bin/python scripts/bayes_regression.py
.venv/bin/python scripts/mlp.py
.venv/bin/python scripts/make_figures.py
```

To rebuild the data from scratch, run `scripts/fetch_humidity.py` and then
`scripts/build_dataset.py` first.

## Data sources

| Data | Source | Notes |
|---|---|---|
| Weather | NOAA GHCN-Daily, station USW00093821 | Temperature, rain, snow, wind. No humidity. |
| Pollen | CDC [Daily Grouped Pollen Concentrations](https://data.cdc.gov/National-Center-for-Environmental-Health/Daily-Grouped-Pollen-Concentrations-Trees-and-Weeds/435k-2whj), from the National Allergy Bureau (NAB) station in Louisville | Counts are grouped as "trees" and "weeds" only. Individual species and grass are withheld under NAB data-sharing rules. 6,206 days with counts. |
| Humidity | [Open-Meteo Historical Weather API](https://open-meteo.com/en/docs/historical-weather-api) | Modeled (ERA5 reanalysis, ~25 km grid), not measured at the airport. Used because NOAA's station data service returned empty files. |

Dataset paper: Manangan et al. (2026), "Pollen Season Timing and
Concentrations in the United States," *Atmosphere* 17(7):635,
[doi:10.3390/atmos17070635](https://doi.org/10.3390/atmos17070635).

## Modeling setup

- **Target:** log(1 + tree pollen grains/m³), March–June only (the tree season).
- **Inputs:** yesterday's log pollen, day of year (and its square), growing
  degree days since Jan 1, average temperature, dew point, mean humidity,
  rain amount, rain-day flag, wind speed, wind direction (sine/cosine).
- **Split by year:** train 2003–2017, test 2018–2022 (576 days). The MLP holds
  out 2015–2017 from its training years to decide when to stop training.
- **Baselines:** the average for that day of year across training years, and
  "same as yesterday".

### Method 1: Bayesian linear regression (`scripts/bayes_regression.py`)

A conjugate Normal-Inverse-Gamma model written in NumPy: a Gaussian prior on
the coefficients that shrinks them toward zero, and an inverse-gamma prior on
the noise variance. The posterior and the Student-t predictive distribution
are computed in closed form, so each forecast comes with a credible interval.

### Method 2: MLP (`scripts/mlp.py`)

Two hidden layers (32 and 16 ReLU units), trained with backpropagation and
Adam on squared error with L2 weight decay. Training stops when the
2015–2017 error stops improving. The gradients were checked against numerical
estimates. Results are averaged over 5 random seeds.

## Results (test seasons 2018–2022)

| Model | RMSE (log) | MAE (log) | R² | Pollen level correct | Within one level |
|---|---|---|---|---|---|
| Day-of-year average | 1.496 | 1.119 | 0.50 | 60.8% | 94.1% |
| Same as yesterday | 1.549 | 1.083 | 0.46 | 62.7% | 95.0% |
| Bayesian regression | 1.264 | 0.936 | 0.64 | 64.1% | 97.6% |
| MLP (5-seed average) | **1.222** | **0.888** | **0.66** | **66.5%** | **98.3%** |

Pollen levels use the CDC tree thresholds: Low under 15, Moderate 15–90,
High 90–1,500, Very High 1,500 and up (grains/m³).

- Both models beat both baselines. The MLP is slightly better than the
  Bayesian model, and better in 4 of the 5 test years.
- The Bayesian 90% prediction intervals contain 89.6% of observed test days,
  and the 50% intervals contain 54.9%, so the uncertainty is well calibrated.
  The intervals are wide, though: the typical 90% interval spans a factor of
  about 48 in pollen count.
- Yesterday's pollen and the time of year do most of the work. Weather adds
  about 3% of explained variation in the Bayesian model. Wind speed has a
  small positive effect and rain has no clear effect.

## Known limitations

- The models use each day's *actual* weather. A real forecast would have only
  a weather forecast, so these results are a best case.
- Temperature, dew point and humidity are strongly related, so their
  separate coefficients in the Bayesian model shouldn't be read on their own.
- 18% of spring days have zero pollen. The models treat these like any other
  value and sometimes predict slightly below zero in late June.
- Only 2 test days reached Very High, so the models' skill on extreme days
  can't really be judged.
- Humidity is modeled, not station-measured.
