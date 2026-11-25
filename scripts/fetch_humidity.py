"""
DS 4420 Project - download daily humidity and dew point for the Louisville
airport (2003-2022) from the Open-Meteo historical weather API (ERA5).
NOAA's daily data doesn't include humidity, so we get it here.
https://open-meteo.com/en/docs/historical-weather-api

Output: data/louisville_humidity_daily.csv
Run from the repo root: .venv/bin/python scripts/fetch_humidity.py
"""

import json
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

# location of the NOAA station (Louisville International Airport)
PARAMS = {
    "latitude": 38.17738,
    "longitude": -85.73077,
    "start_date": "2003-01-01",
    "end_date": "2022-12-31",
    "daily": ",".join([
        "relative_humidity_2m_mean",
        "relative_humidity_2m_min",
        "relative_humidity_2m_max",
        "dew_point_2m_mean",
    ]),
    "timezone": "America/New_York",
}
URL = "https://archive-api.open-meteo.com/v1/archive?" + urllib.parse.urlencode(PARAMS)


def main():
    with urllib.request.urlopen(URL, timeout=120) as resp:
        daily = json.load(resp)["daily"]
    df = pd.DataFrame({
        "date": daily["time"],
        "rh_mean_pct": daily["relative_humidity_2m_mean"],
        "rh_min_pct": daily["relative_humidity_2m_min"],
        "rh_max_pct": daily["relative_humidity_2m_max"],
        # convert C to F to match the NOAA data
        "dewpoint_mean_f": [None if c is None else round(c * 9 / 5 + 32, 1)
                            for c in daily["dew_point_2m_mean"]],
    })
    out = ROOT / "data" / "louisville_humidity_daily.csv"
    df.to_csv(out, index=False)
    print(f"wrote {out.relative_to(ROOT)}: {len(df)} rows")


if __name__ == "__main__":
    main()
