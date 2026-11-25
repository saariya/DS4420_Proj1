"""
DS 4420 Project - figures for the report and poster.

  fig_season.png         average tree pollen season by day of year
  fig_forecast_2019.png  2019 forecast with 90% prediction intervals
  fig_drivers.png        posterior coefficients with 95% posterior intervals
  fig_accuracy.png       R^2 for both models and both baselines

Run from the repo root: .venv/bin/python scripts/make_figures.py
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import mlp
from bayes_regression import (FEATURES, TEST_START_YEAR, BayesLinReg, design,
                              load)

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "figures"

# colors (blue = Bayesian model, orange = MLP, grays for everything else)
BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#a3a29d", "#e6e5e1"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.edgecolor": MUTED,
    "axes.labelcolor": INK2,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "axes.titlelocation": "left",
    "axes.titlepad": 22,
    "xtick.color": INK2,
    "ytick.color": INK2,
    "xtick.major.size": 0,
    "ytick.major.size": 0,
    "axes.grid": False,
    "grid.color": GRID,
    "grid.linewidth": 0.8,
    "figure.dpi": 100,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
})

LABELS = {
    "log_tree_lag1": "Yesterday's pollen",
    "doy": "Day of year",
    "doy_sq": "Day of year²",
    "gdd_cum": "Growing degree days",
    "tavg_f": "Temperature",
    "dewpoint_mean_f": "Dew point",
    "rh_mean_pct": "Humidity",
    "prcp_in": "Rainfall",
    "rain": "Rain day",
    "wind_avg_mph": "Wind speed",
    "wind_dir_sin": "Wind direction (E–W)",
    "wind_dir_cos": "Wind direction (N–S)",
}
POLLEN_TICKS = [0, 10, 100, 1000]


def subtitle(ax, text):
    ax.text(0, 1.02, text, transform=ax.transAxes, color=INK2, fontsize=9.5, va="bottom")


def pollen_axis(ax):
    ax.set_yscale("symlog", linthresh=10)
    ax.set_yticks(POLLEN_TICKS, ["0", "10", "100", "1,000"])
    ax.set_ylim(0, 4000)
    ax.yaxis.grid(True)
    ax.set_axisbelow(True)
    ax.spines["left"].set_visible(False)


def fig_season():
    d = pd.read_csv(ROOT / "data" / "louisville_merged.csv", parse_dates=["date"])
    d = d[d["has_pollen"]]
    by = d.groupby("doy")["tree_pollen"]
    q = pd.DataFrame({"q25": by.quantile(0.25), "med": by.median(), "q75": by.quantile(0.75)})
    q = q.rolling(7, center=True, min_periods=1).mean().loc[15:230]
    x = pd.Timestamp("2021-01-01") + pd.to_timedelta(q.index - 1, unit="D")  # any non-leap year

    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    ax.axvspan(pd.Timestamp("2021-03-01"), pd.Timestamp("2021-06-30"), color=GRID, alpha=0.5, lw=0)
    ax.fill_between(x, q["q25"], q["q75"], color=BLUE, alpha=0.18, lw=0)
    ax.plot(x, q["med"], color=BLUE, lw=2)
    pollen_axis(ax)
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    ax.set_ylabel("Tree pollen (grains/m³)")

    peak = q["med"].idxmax()
    px = pd.Timestamp("2021-01-01") + pd.Timedelta(days=int(peak) - 1)
    ax.annotate(f"Median peaks ~{px:%b} {px.day}\nat {q['med'].max():,.0f} grains/m³",
                xy=(px, q["med"].max()), xytext=(px + pd.Timedelta(days=30), 1100),
                color=INK, fontsize=9, arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8))
    ax.text(pd.Timestamp("2021-05-20"), q.loc[140, "q75"] * 1.6, "middle 50% of years",
            color=BLUE, fontsize=9)
    ax.text(pd.Timestamp("2021-03-03"), 2400, "Modeled season (Mar–Jun)", color=INK2, fontsize=9)
    ax.set_title("Louisville's tree pollen season peaks in April")
    subtitle(ax, "Daily tree pollen by calendar day, 2003–2022 (7-day smoothed)")
    fig.savefig(FIG / "fig_season.png")
    plt.close(fig)


def fig_forecast(test, yte, mean, lo, hi, coverage):
    show = test["year"].to_numpy() == 2019
    dates = test["date"].to_numpy()[show]
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    ax.fill_between(dates, np.expm1(lo[show]).clip(0), np.expm1(hi[show]),
                    color=BLUE, alpha=0.15, lw=0)
    ax.plot(dates, np.expm1(mean[show]).clip(0), color=BLUE, lw=1.6)
    ax.plot(dates, np.expm1(yte[show]), "o", ms=3.2, color=INK, mec="white", mew=0.5)
    pollen_axis(ax)
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    ax.set_ylabel("Tree pollen (grains/m³)")

    # label the lines directly instead of using a legend
    ax.text(pd.Timestamp("2019-05-27"), 260, "90% prediction\ninterval", color=BLUE, fontsize=9)
    ax.annotate("observed count", xy=(pd.Timestamp("2019-03-14"), 1040),
                xytext=(pd.Timestamp("2019-03-19"), 2300), color=INK, fontsize=9,
                arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8))
    ax.annotate("forecast", xy=(pd.Timestamp("2019-06-05"), 14),
                xytext=(pd.Timestamp("2019-06-12"), 70), color=BLUE, fontsize=9,
                arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8))
    ax.set_title("The forecast tracks a season it never saw", pad=34)
    subtitle(ax, f"2019 test season, forecast one day ahead. Across all test years,\n"
                 f"{coverage:.0%} of days fell inside the 90% interval.")
    fig.savefig(FIG / "fig_forecast_2019.png")
    plt.close(fig)


def fig_drivers(model):
    clo, chi = model.coef_interval()
    m, clo, chi = model.w_n[1:], clo[1:], chi[1:]  # skip the intercept
    order = np.argsort(np.abs(m))
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    for row, j in enumerate(order):
        clear = not (clo[j] < 0 < chi[j])
        c = BLUE if clear else MUTED
        ax.plot([clo[j], chi[j]], [row, row], color=c, lw=2, solid_capstyle="round")
        ax.plot(m[j], row, "o", color=c, ms=6, mec="white", mew=1)
    ax.axvline(0, color=MUTED, lw=0.8)
    ax.set_yticks(range(len(order)), [LABELS[FEATURES[j]] for j in order])
    ax.tick_params(axis="y", colors=INK)
    ax.spines["left"].set_visible(False)
    ax.xaxis.grid(True)
    ax.set_axisbelow(True)
    ax.set_xlabel("Effect on log pollen per 1 standard deviation (95% posterior interval)")
    ax.text(0.99, 0.02, "gray = interval includes 0", transform=ax.transAxes,
            ha="right", color=INK2, fontsize=9)
    ax.set_title("Yesterday's pollen and the calendar drive the forecast", pad=34)
    subtitle(ax, "Bayesian regression coefficients. Weather effects are small; day of year\n"
                 "and its square together trace the season's rise and fall.")
    fig.savefig(FIG / "fig_drivers.png")
    plt.close(fig)


def fig_accuracy(r2):
    names = list(r2)
    vals = np.array([r2[n] for n in names])
    colors = [MUTED, MUTED, BLUE, ORANGE]
    fig, ax = plt.subplots(figsize=(7.0, 2.6))
    y = np.arange(len(names))[::-1]
    for yi, v, c in zip(y, vals, colors):
        ax.plot([0, v], [yi, yi], color=c, lw=2, solid_capstyle="round")
        ax.plot(v, yi, "o", color=c, ms=8, mec="white", mew=1)
        ax.text(v + 0.015, yi, f"{v:.2f}", va="center", color=INK, fontsize=10)
    ax.set_yticks(y, names)
    ax.tick_params(axis="y", colors=INK)
    ax.set_xlim(0, 0.8)
    ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8])
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("R²: share of day-to-day variation explained (test years 2018–2022)")
    ax.set_title("Both models beat simple rules of thumb")
    subtitle(ax, "Higher is better. 576 test days the models never saw in training.")
    fig.savefig(FIG / "fig_accuracy.png")
    plt.close(fig)


def main():
    FIG.mkdir(exist_ok=True)
    fig_season()

    d = load()
    train, test = d[d["year"] < TEST_START_YEAR], d[d["year"] >= TEST_START_YEAR]
    mu, sd = train[FEATURES].mean(), train[FEATURES].std()
    model = BayesLinReg().fit(design(train, mu, sd), train["log_tree"].to_numpy())
    yte = test["log_tree"].to_numpy()
    mean, lo, hi = model.predict(design(test, mu, sd))
    coverage = np.mean((yte >= lo) & (yte <= hi))
    fig_forecast(test, yte, mean, lo, hi, coverage)
    fig_drivers(model)

    # baselines and MLP, same as in bayes_regression.py and mlp.py
    clim = train.groupby("doy")["log_tree"].mean().rolling(7, center=True, min_periods=1).mean()
    clim_pred = test["doy"].map(clim).fillna(train["log_tree"].mean()).to_numpy()
    persist = test["log_tree_lag1"].to_numpy()
    fit = d[d["year"] < mlp.VAL_START_YEAR]
    val = d[(d["year"] >= mlp.VAL_START_YEAR) & (d["year"] < TEST_START_YEAR)]
    mu2, sd2 = fit[FEATURES].mean(), fit[FEATURES].std()
    X = lambda df: design(df, mu2, sd2)[:, 1:]
    ens = np.mean([mlp.train(X(fit), fit["log_tree"].to_numpy(), X(val),
                             val["log_tree"].to_numpy(), s)[0].predict(X(test))
                   for s in mlp.SEEDS], axis=0)
    r2 = lambda p: 1 - np.mean((yte - p) ** 2) / yte.var()
    fig_accuracy({"Same as yesterday": r2(persist),
                  "Day-of-year average": r2(clim_pred),
                  "Bayesian regression": r2(mean),
                  "Neural network (MLP)": r2(ens)})
    print("saved figures/fig_season.png, fig_forecast_2019.png, fig_drivers.png, fig_accuracy.png")


if __name__ == "__main__":
    main()
