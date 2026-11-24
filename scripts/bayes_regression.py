"""Prototype: Bayesian linear regression for daily tree pollen in Louisville, KY.

Implemented by hand in NumPy (no sklearn/statsmodels), using the conjugate
Normal-Inverse-Gamma model:

    y | X, beta, sigma^2  ~  N(X beta, sigma^2 I)
    beta | sigma^2        ~  N(0, sigma^2 Lambda0^-1)
    sigma^2               ~  InvGamma(a0, b0)

y is log(1 + tree pollen grains/m^3), restricted to the March-June season.
Each row predicts today's pollen from yesterday's pollen, the time of year,
and today's weather (in a real forecast that weather would itself be a forecast).

Train: 2003-2017 seasons. Test: 2018-2022 seasons.
Run from the repo root:  .venv/bin/python scripts/bayes_regression.py
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "figures"

TEST_START_YEAR = 2018
FEATURES = [
    "log_tree_lag1",          # yesterday's pollen (autoregressive term)
    "doy", "doy_sq",          # season shape: rise then fall
    "gdd_cum",                # accumulated warmth since Jan 1
    "tavg_f", "dewpoint_mean_f", "rh_mean_pct",
    "prcp_in", "rain", "wind_avg_mph",
    "wind_dir_sin", "wind_dir_cos",
]
PRIOR_PRECISION = 1.0       # ridge-like shrinkage on standardized coefficients
INTERCEPT_PRECISION = 1e-6  # essentially flat prior on the intercept
A0, B0 = 1.0, 1.0           # weak InvGamma prior on the noise variance


# ---------------------------------------------------------------- data

def load() -> pd.DataFrame:
    d = pd.read_csv(ROOT / "data" / "louisville_merged.csv", parse_dates=["date"])
    d = d[d["has_pollen"] & d["month"].between(3, 6)].copy()
    d["doy_sq"] = d["doy"] ** 2
    return d.dropna(subset=FEATURES + ["log_tree"])


def design(df: pd.DataFrame, mu: pd.Series, sd: pd.Series) -> np.ndarray:
    z = (df[FEATURES] - mu) / sd  # standardize with *training* statistics
    return np.column_stack([np.ones(len(df)), z.to_numpy()])


# ---------------------------------------------------------------- model

class BayesLinReg:
    def fit(self, X: np.ndarray, y: np.ndarray) -> "BayesLinReg":
        n, p = X.shape
        lam0 = np.full(p, PRIOR_PRECISION)
        lam0[0] = INTERCEPT_PRECISION
        prec_n = X.T @ X + np.diag(lam0)               # posterior precision (/sigma^2)
        self.V = np.linalg.inv(prec_n)
        self.m = self.V @ (X.T @ y)                    # posterior mean (prior mean 0)
        self.a = A0 + n / 2
        self.b = B0 + 0.5 * (y @ y - self.m @ prec_n @ self.m)
        return self

    def coef_interval(self, level: float = 0.95):
        """Marginal posterior of each beta_j is Student-t(2a, m_j, b/a * V_jj)."""
        scale = np.sqrt(self.b / self.a * np.diag(self.V))
        t = stats.t.ppf(0.5 + level / 2, df=2 * self.a)
        return self.m - t * scale, self.m + t * scale

    def predict(self, X: np.ndarray, level: float = 0.9):
        """Posterior predictive is Student-t(2a, X m, b/a * (1 + x V x'))."""
        mean = X @ self.m
        scale = np.sqrt(self.b / self.a * (1 + np.einsum("ij,jk,ik->i", X, self.V, X)))
        t = stats.t.ppf(0.5 + level / 2, df=2 * self.a)
        return mean, mean - t * scale, mean + t * scale


# ---------------------------------------------------------------- evaluation

def rmse(y, yhat) -> float:
    return float(np.sqrt(np.mean((y - yhat) ** 2)))


def pollen_level(log_counts: np.ndarray) -> np.ndarray:
    """CDC grouped-tree levels: 0 Low <15, 1 Moderate 15-90, 2 High 90-1500, 3 Very High."""
    g = np.expm1(log_counts) + 1e-9  # so an observed count of exactly 15 or 90 isn't nudged below
    return np.select([g >= 1500, g >= 90, g >= 15], [3, 2, 1], 0)


def print_metrics(y: np.ndarray, preds: dict) -> None:
    """Table of test metrics shared by both models (all on log(1 + grains/m^3))."""
    lv = pollen_level(y)
    print(f"  {'model':32s} {'RMSE':>6s} {'MAE':>6s} {'R2':>6s} {'level':>7s} "
          f"{'within1':>8s} {'high recall':>12s}")
    for name, p in preds.items():
        lp = pollen_level(p)
        print(f"  {name:32s} {rmse(y, p):6.3f} {np.mean(np.abs(y - p)):6.3f} "
              f"{1 - np.mean((y - p) ** 2) / y.var():6.3f} {np.mean(lp == lv):7.1%} "
              f"{np.mean(np.abs(lp - lv) <= 1):8.1%} {np.mean(lp[lv >= 2] >= 2):12.1%}")
    print("  (level = CDC pollen level correct; high recall = share of High/Very High "
          "days predicted High or above)")


def main() -> None:
    d = load()
    train, test = d[d["year"] < TEST_START_YEAR], d[d["year"] >= TEST_START_YEAR]
    mu, sd = train[FEATURES].mean(), train[FEATURES].std()
    Xtr, Xte = design(train, mu, sd), design(test, mu, sd)
    ytr, yte = train["log_tree"].to_numpy(), test["log_tree"].to_numpy()

    model = BayesLinReg().fit(Xtr, ytr)
    mean, lo, hi = model.predict(Xte)

    # Baselines: persistence (tomorrow = today) and day-of-year climatology.
    clim = train.groupby("doy")["log_tree"].mean()
    clim = clim.rolling(7, center=True, min_periods=1).mean()
    clim_pred = test["doy"].map(clim).fillna(ytr.mean()).to_numpy()
    persist = test["log_tree_lag1"].to_numpy()
    lag_only = BayesLinReg().fit(Xtr[:, :2], ytr).predict(Xte[:, :2])[0]

    print(f"train days {len(train)}  test days {len(test)}\n")
    print("Test metrics on log(1 + grains/m^3):")
    print_metrics(yte, {"climatology (day of year)": clim_pred,
                        "persistence (yesterday)": persist,
                        "Bayes, lag only": lag_only,
                        "Bayes, lag + season + weather": mean})
    _, lo50, hi50 = model.predict(Xte, level=0.5)
    print(f"\npredictive interval coverage on test: "
          f"50% -> {np.mean((yte >= lo50) & (yte <= hi50)):.1%}, "
          f"90% -> {np.mean((yte >= lo) & (yte <= hi)):.1%}")
    print(f"median 90% interval spans a factor of {np.exp(np.median(hi - lo)):.0f} in pollen count")
    print(f"typical miss (median): factor of {np.exp(np.median(np.abs(yte - mean))):.2f}")
    print(f"posterior mean noise sd: {np.sqrt(model.b / (model.a - 1)):.3f}\n")

    clo, chi = model.coef_interval()
    names = ["intercept"] + FEATURES
    print(f"{'coefficient (per 1 sd)':24s} {'mean':>7s}   95% credible interval")
    for n, m, l, h in zip(names, model.m, clo, chi):
        flag = "" if l < 0 < h else "  *"
        print(f"  {n:22s} {m:+7.3f}   [{l:+.3f}, {h:+.3f}]{flag}")
    print("  (* = interval excludes 0)")

    # Prior sensitivity: refit with weaker and stronger shrinkage on the coefficients.
    global PRIOR_PRECISION
    default = PRIOR_PRECISION
    print("\nprior sensitivity (coefficient prior precision -> test RMSE, 90% coverage):")
    for lam in [0.01, 1, 10, 100, 1000]:
        PRIOR_PRECISION = lam
        p, l90, h90 = BayesLinReg().fit(Xtr, ytr).predict(Xte)
        print(f"  {lam:>7g}: RMSE {rmse(yte, p):.3f}, coverage {np.mean((yte >= l90) & (yte <= h90)):.1%}")
    PRIOR_PRECISION = default

    plot(test, yte, mean, lo, hi, names[1:], model.m[1:], clo[1:], chi[1:])


def plot(test, yte, mean, lo, hi, names, m, clo, chi) -> None:
    FIG.mkdir(exist_ok=True)

    yr = test["year"].to_numpy()
    show = yr == 2019  # most complete test season
    fig, ax = plt.subplots(figsize=(9, 4))
    dates = test["date"].to_numpy()[show]
    ax.fill_between(dates, np.expm1(lo[show]).clip(0), np.expm1(hi[show]),
                    color="tab:blue", alpha=0.2, label="90% predictive interval")
    ax.plot(dates, np.expm1(mean[show]), color="tab:blue", label="posterior mean")
    ax.plot(dates, np.expm1(yte[show]), "k.", ms=4, label="observed")
    ax.set_yscale("symlog", linthresh=10)
    ax.set_ylabel("tree pollen (grains/m³)")
    ax.set_title("Louisville tree pollen, 2019 test season: one-day-ahead forecast")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(FIG / "forecast_2019.png", dpi=150)

    order = np.argsort(m)
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.errorbar(m[order], np.arange(len(m)),
                xerr=[m[order] - clo[order], chi[order] - m[order]],
                fmt="o", color="tab:blue", capsize=3)
    ax.axvline(0, color="grey", lw=1)
    ax.set_yticks(np.arange(len(m)), [names[i] for i in order])
    ax.set_xlabel("effect on log pollen per 1 sd (95% credible interval)")
    ax.set_title("Posterior coefficients")
    fig.tight_layout()
    fig.savefig(FIG / "coefficients.png", dpi=150)
    print(f"\nsaved figures/forecast_2019.png and figures/coefficients.png")


if __name__ == "__main__":
    main()
