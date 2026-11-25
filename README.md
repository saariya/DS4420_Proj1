# DS 4420 Project: Pollen Monitoring

Two parts of pollen monitoring, each with a different method:

1. **Identification:** a CNN that classifies six tree pollen species from microscope images.
2. **Forecasting:** a Bayesian linear regression (written by hand in NumPy) that forecasts daily tree pollen in Louisville, KY, from the previous day's count, the time of year and the weather. An MLP is used as a comparison.

The two parts use separate datasets.

## Files

```
data/                     all data used (images for the CNN are downloaded separately, see below)
scripts/
  pollen_cnn.ipynb        CNN: cropping, splitting by photo, training, evaluation
  pollen_forecast.ipynb   Bayesian regression, step by step
  bayes_regression.py     Bayesian regression and baselines (prints all forecasting results)
  mlp.py                  MLP comparison
  make_figures.py         figures used in the report
  fetch_humidity.py       downloads the humidity data
  build_dataset.py        joins weather, humidity and pollen into data/louisville_merged.csv
figures/                  saved figures
```

## Setup

Tested with Python 3.12.

```bash
python -m venv .venv
.venv/bin/pip install -r requirements.txt
```

## Forecasting

The data is already in `data/`, so the models can be run directly from the repo root:

```bash
.venv/bin/python scripts/bayes_regression.py
.venv/bin/python scripts/mlp.py
.venv/bin/python scripts/make_figures.py
```

`scripts/pollen_forecast.ipynb` shows the same Bayesian model step by step (run it from `scripts/`).

To rebuild the data from scratch, run `scripts/fetch_humidity.py` and then `scripts/build_dataset.py`.

## CNN

1. Download the POLLEN20L-det images from [Kaggle](https://www.kaggle.com/datasets/nataliakhanzhina/pollen20ldet) into `data/images/`.
2. Install the extra packages: `.venv/bin/pip install tensorflow keras pillow scikit-learn`
3. Run `scripts/pollen_cnn.ipynb` from the `scripts/` folder. It saves `confusion_matrix.png` and `pollen_cnn.keras` there.

## Data sources

- **Pollen images:** POLLEN20L-det (Kaggle), 7,744 labeled grains across 20 types.
- **Pollen counts:** CDC [Daily Grouped Pollen Concentrations](https://data.cdc.gov/d/435k-2whj), Louisville NAB station, 2003–2022. Trees are grouped together in the public data.
- **Weather:** NOAA daily data, Louisville International Airport (USW00093821).
- **Humidity and dew point:** [Open-Meteo](https://open-meteo.com/en/docs/historical-weather-api) (ERA5), since NOAA's daily data doesn't include them.

## Results

**CNN** (484 test grains): 98.1% accuracy and macro F1 of 0.976, vs. 24.4% for always guessing the most common species.

**Forecasting** (576 test days, 2018–2022, on log(1 + grains/m³)):

| Model | RMSE | R² |
|---|---|---|
| Day-of-year average | 1.496 | 0.497 |
| Same as yesterday | 1.549 | 0.461 |
| Bayesian regression | 1.264 | 0.641 |
| MLP (average of 5 runs) | 1.222 | 0.664 |

The Bayesian model's 90% prediction intervals contained 89.6% of test days.
