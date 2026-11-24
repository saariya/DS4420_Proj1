# DS4420_Proj1: Pollen monitoring

Pollen monitoring has two parts: identifying which pollen is in the air and
predicting how much there will be. This repo covers both:

1. **Identification:** a convolutional neural network (CNN) that classifies
   tree pollen species from cropped microscope images of single grains.
2. **Prediction:** models that forecast daily **tree pollen in Louisville, KY**
   from the previous day's count, the time of year and the weather.

The two parts are developed and evaluated separately. The CNN is trained on
microscope images and the forecasting models on daily counts from one
monitoring station, so the CNN's output is not yet an input to the forecasts.

## Repository layout

```
data/
  bboxes.csv                    Grain bounding boxes and species labels (POLLEN20L-det), no header row
  class_map.csv                 Class ID to species name mapping
  images/                       Microscope photos named in bboxes.csv
  weather_data.csv              NOAA daily weather, Louisville airport (USW00093821), 2003-2022
  louisville_pollen_daily.csv   CDC/NAB daily tree and weed pollen counts (grains/m³), 2003-2022
  louisville_humidity_daily.csv Daily relative humidity and dew point (Open-Meteo, ERA5 reanalysis)
  louisville_merged.csv         All of the above joined by date, plus engineered features
scripts/
  pollen_cnn.ipynb              Pollen species CNN: cropping, splitting, training, evaluation
  pollen_cnn.keras              Trained CNN saved by the notebook
  confusion_matrix.png          CNN confusion matrix on the held-out test photos
  fetch_humidity.py             Downloads the humidity file
  build_dataset.py              Builds louisville_merged.csv
  bayes_regression.py           Forecasting method 1: Bayesian linear regression (NumPy only)
  mlp.py                        Forecasting method 2: multilayer perceptron (NumPy only)
figures/
  forecast_2019.png             Bayesian forecast with 90% intervals, 2019 season
  coefficients.png              Posterior effect of each input
  mlp_vs_bayes_2019.png         Both models' forecasts, 2019 season
  mlp_training_curve.png        MLP train/validation error by epoch
pollen_cnn.keras                Earlier saved copy of the CNN (the one in scripts/ is newer)
requirements.txt
```

## Setup and running

Tested with Python 3.12. From the repo root:

```bash
python -m venv .venv
.venv/bin/pip install -r requirements.txt
```

### Forecasting models

The data files are already committed, so you can go straight to the models:

```bash
.venv/bin/python scripts/bayes_regression.py
.venv/bin/python scripts/mlp.py
```

To rebuild the data from scratch, run `scripts/fetch_humidity.py` and then
`scripts/build_dataset.py` first.

### Pollen CNN

The CNN needs `keras`, `tensorflow`, `pillow` and `scikit-learn` in addition
to the packages above. Open `scripts/pollen_cnn.ipynb` and run it from the
`scripts/` folder, because the notebook reads `../data/bboxes.csv` and
`../data/images/` by relative path. It writes `confusion_matrix.png` and
`pollen_cnn.keras` into `scripts/`.

## Data sources

| Data | Source | Notes |
|---|---|---|
| Pollen images | POLLEN20L-det (Kaggle) | Microscope photos with a bounding box and species label for each grain. 20 classes, 7,744 grains in 2,412 image files from 1,787 original photos. |
| Weather | NOAA GHCN-Daily, station USW00093821 | Temperature, rain, snow, wind. No humidity. |
| Pollen counts | CDC [Daily Grouped Pollen Concentrations](https://data.cdc.gov/National-Center-for-Environmental-Health/Daily-Grouped-Pollen-Concentrations-Trees-and-Weeds/435k-2whj), from the National Allergy Bureau (NAB) station in Louisville | Counts are grouped as "trees" and "weeds" only. Individual species and grass are withheld under NAB data-sharing rules. 6,206 days with counts. |
| Humidity | [Open-Meteo Historical Weather API](https://open-meteo.com/en/docs/historical-weather-api) | Modeled (ERA5 reanalysis, ~25 km grid), not measured at the airport. Used because NOAA's station data service returned empty files. |

Dataset paper: Manangan et al. (2026), "Pollen Season Timing and
Concentrations in the United States," *Atmosphere* 17(7):635,
[doi:10.3390/atmos17070635](https://doi.org/10.3390/atmos17070635).

## Part 1: Pollen species CNN (`scripts/pollen_cnn.ipynb`)

**Question:** can a small CNN identify tree pollen species from cropped
microscope images?

### Data preparation

- **Classes:** the six tree species maple, willow, birch, alder, linden and
  pine. The other 14 classes (weeds, grasses and similar) are excluded. Hazel
  is left out because it has only 77 grains.
- **Cropping:** each grain is cropped from its photo using its bounding box,
  padded to a square with black so grains are not stretched, and resized to
  64×64 RGB. This gives 3,680 crops.
- **Class imbalance:** maple has 1,159 grains and pine 230. Rarer species get
  larger class weights during training (pine ×2.76, linden ×1.90, maple
  ×0.51), and results are reported per class rather than by accuracy alone.

### Split by original photo

File names look like Roboflow exports (`<photo>.rf.<hash>.jpg`), so the same
photo can appear several times as augmented copies. A random split by grain
could place near-identical copies in both train and test and inflate accuracy.
Instead the data is split by **original photo** (70% train, 15% validation,
15% test, seed 42), and the notebook asserts that no photo appears in more
than one split.

| Class | Train | Val | Test |
|---|---|---|---|
| maple | 873 | 168 | 118 |
| willow | 561 | 144 | 111 |
| birch | 501 | 37 | 128 |
| alder | 352 | 63 | 55 |
| linden | 236 | 59 | 44 |
| pine | 162 | 40 | 28 |

### Model

Input (64×64×3) → rescale to 0–1 → random flip and random rotation
(augmentation, active only during training, since pollen grains have no fixed
orientation) → three blocks of Conv2D (16, 32, 64 filters, 3×3, ReLU, same
padding) + MaxPooling → Flatten → Dense(64, ReLU) → Dropout(0.4) → Dense(6,
softmax). It has about 286,000 trainable parameters.

Training uses Adam (learning rate 1e-3) with sparse categorical
cross-entropy, batch size 32, up to 40 epochs, and early stopping on
validation loss (patience 6, best weights restored).

### Results (held-out test photos, 484 grains)

| Metric | Value |
|---|---|
| Test accuracy | **98.1%** |
| Majority-class baseline (always predict maple) | 24.4% |
| Macro F1 | **0.976** |

| Class | Precision | Recall | F1 | Test grains |
|---|---|---|---|---|
| maple | 0.99 | 0.99 | 0.99 | 118 |
| willow | 1.00 | 0.95 | 0.98 | 111 |
| birch | 1.00 | 1.00 | 1.00 | 128 |
| alder | 0.88 | 0.96 | 0.92 | 55 |
| linden | 1.00 | 1.00 | 1.00 | 44 |
| pine | 0.96 | 0.96 | 0.96 | 28 |

The confusion matrix is saved as `scripts/confusion_matrix.png`. Alder is the
weakest class, mostly because other grains get predicted as alder.

### CNN limitations

- The test set is small (28 pine and 44 linden grains), so one misclassified
  grain moves a per-class score noticeably.
- Only six tree species are covered, and all images come from one dataset.
  Performance on other microscopes, stains or sample preparation is untested.
- The model classifies single, already-cropped grains. It does not find grains
  in a full photo (the boxes come from the dataset's annotations).

## Part 2: Pollen forecasting

### Modeling setup

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

### Results (test seasons 2018–2022)

| Model | RMSE (log) | MAE (log) | R² | Pollen level correct | Within one level |
|---|---|---|---|---|---|
| Day-of-year average | 1.496 | 1.119 | 0.50 | 60.8% | 94.1% |
| Same as yesterday | 1.549 | 1.083 | 0.46 | 62.0% | 94.8% |
| Bayesian regression | 1.264 | 0.936 | 0.64 | 64.1% | 97.6% |
| MLP (5-seed average) | **1.222** | **0.888** | **0.66** | **66.5%** | **98.3%** |

Pollen levels use the CDC tree thresholds: Low under 15, Moderate 15–90,
High 90–1,500, Very High 1,500 and up (grains/m³).

- Both models beat both baselines. The MLP is slightly better than the
  Bayesian model, and better in 4 of the 5 test years.
- The Bayesian 90% prediction intervals contain 89.6% of observed test days,
  and the 50% intervals contain 54.9%, so the uncertainty is well calibrated.
  The intervals are wide, though: the typical 90% interval spans a factor of
  about 50 in pollen count.
- Yesterday's pollen and the time of year do most of the work. Weather adds
  about 3% of explained variation in the Bayesian model. Wind speed has a
  small positive effect and rain has no clear effect.

### Forecasting limitations

- The models use each day's *actual* weather. A real forecast would have only
  a weather forecast, so these results are a best case.
- Temperature, dew point and humidity are strongly related, so their
  separate coefficients in the Bayesian model shouldn't be read on their own.
- 18% of spring days have zero pollen. The models treat these like any other
  value and sometimes predict slightly below zero in late June.
- Only 2 test days reached Very High, so the models' skill on extreme days
  can't really be judged.
- Humidity is modeled, not station-measured.

## Future work

The motivation for this project is to get closer to showing pollen levels in
the weather apps people already use. The next step is to connect the two
parts, for example by using species-level identification to separate which
trees are driving the daily counts, and by replacing observed weather with
weather forecasts in the prediction models.
