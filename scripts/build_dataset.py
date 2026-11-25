"""
DS 4420 Project - combine the Louisville weather (NOAA), humidity (Open-Meteo)
and pollen (CDC/NAB) data into one table with one row per day, 2003-2022.

We keep every day (even without a pollen count) so the lag and running-total
features don't skip days. Output: data/louisville_merged.csv

Run from the repo root: .venv/bin/python scripts/build_dataset.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

# NOAA weather flags we keep (1 = happened that day, blank = didn't)
FLAGS = {
    "WT01": "fog",
    "WT03": "thunder",
    "WT05": "hail",
    "WT08": "haze",
    "WT16": "rain",
    "WT18": "snow_flag",
}
GDD_BASE_F = 41.0  # base temperature for growing degree days (5 C)


def load_weather():
    w = pd.read_csv(DATA / "weather_data.csv", encoding="utf-8-sig")
    w["date"] = pd.to_datetime(w["DATE"], format="%m/%d/%Y")
    out = pd.DataFrame({
        "date": w["date"],
        "tmax_f": w["TMAX"],
        "tmin_f": w["TMIN"],
        # TAVG is mostly missing, so use the average of max and min instead
        "tavg_f": (w["TMAX"] + w["TMIN"]) / 2,
        "prcp_in": w["PRCP"],
        "snow_in": w["SNOW"],
        "snow_depth_in": w["SNWD"],
        "wind_avg_mph": w["AWND"],
        "gust2_mph": w["WSF2"],
        "gust5_mph": w["WSF5"],
    })
    # wind direction is circular, so use sin/cos (350 and 10 degrees end up close)
    rad = np.deg2rad(w["WDF2"])
    out["wind_dir_sin"] = np.sin(rad)
    out["wind_dir_cos"] = np.cos(rad)
    for code, name in FLAGS.items():
        out[name] = w[code].fillna(0).astype(int)
    return out


def load_pollen():
    p = pd.read_csv(DATA / "louisville_pollen_daily.csv", parse_dates=["date"])
    wide = p.pivot(index="date", columns="pollen_type",
                   values=["pollen_count", "pollen_count_class"])
    return pd.DataFrame({
        "tree_pollen": wide[("pollen_count", "GroupedTrees")].astype(float),
        "tree_class": wide[("pollen_count_class", "GroupedTrees")],
        "weed_pollen": wide[("pollen_count", "GroupedWeeds")].astype(float),
        "weed_class": wide[("pollen_count_class", "GroupedWeeds")],
    }).reset_index()


def load_humidity():
    # made by fetch_humidity.py (from a weather model, not the airport sensors)
    return pd.read_csv(DATA / "louisville_humidity_daily.csv", parse_dates=["date"])


def add_features(df):
    df = df.sort_values("date").reset_index(drop=True)
    # make sure there are no missing dates, since the lags assume consecutive days
    assert (df["date"].diff().dropna() == pd.Timedelta(days=1)).all()

    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["doy"] = df["date"].dt.day_of_year
    df["doy_sin"] = np.sin(2 * np.pi * df["doy"] / 365.25)
    df["doy_cos"] = np.cos(2 * np.pi * df["doy"] / 365.25)

    # growing degree days added up since Jan 1
    gdd = (df["tavg_f"] - GDD_BASE_F).clip(lower=0)
    df["gdd_cum"] = gdd.groupby(df["year"]).cumsum()

    # yesterday's weather and 3-day windows
    df["prcp_lag1"] = df["prcp_in"].shift(1)
    df["tmax_lag1"] = df["tmax_f"].shift(1)
    df["rh_mean_lag1"] = df["rh_mean_pct"].shift(1)
    df["tavg_roll3"] = df["tavg_f"].rolling(3).mean()
    df["prcp_roll3"] = df["prcp_in"].rolling(3).sum()

    # log pollen (the target) and lagged pollen
    df["log_tree"] = np.log1p(df["tree_pollen"])
    df["log_weed"] = np.log1p(df["weed_pollen"])
    df["log_tree_lag1"] = df["log_tree"].shift(1)
    df["log_tree_lag2"] = df["log_tree"].shift(2)
    df["has_pollen"] = df["tree_pollen"].notna()
    return df


def main():
    df = (load_weather()
          .merge(load_humidity(), on="date", how="left", validate="1:1")
          .merge(load_pollen(), on="date", how="left", validate="1:1"))
    df = add_features(df)
    out = DATA / "louisville_merged.csv"
    df.to_csv(out, index=False)
    print(f"wrote {out.relative_to(ROOT)}: {len(df)} rows, {df.shape[1]} columns, "
          f"{int(df['has_pollen'].sum())} with pollen")


if __name__ == "__main__":
    main()
