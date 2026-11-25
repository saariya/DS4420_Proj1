"""
DS 4420 Project - Bayesian linear regression for daily tree pollen in Louisville, KY.

Model (Normal-Inverse-Gamma prior):
    y | w, sigma^2, X ~ N(Xw, sigma^2 I)
    w | sigma^2       ~ N(0, sigma^2 Sigma_0)
    sigma^2           ~ IG(alpha_0, beta_0)

y is log(1 + tree pollen grains/m^3) for March-June.
Train on 2003-2017, test on 2018-2022.

Run from the repo root: .venv/bin/python scripts/bayes_regression.py
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
    "log_tree_lag1",        # yesterday's pollen
    "doy", "doy_sq",        # day of year and its square (season shape)
    "gdd_cum",              # growing degree days since Jan 1
    "tavg_f", "dewpoint_mean_f", "rh_mean_pct",
    "prcp_in", "rain", "wind_avg_mph",
    "wind_dir_sin", "wind_dir_cos",
]

# prior hyperparameters
PRIOR_PRECISION = 1.0       # diagonal of Sigma_0^-1 for the standardized predictors
INTERCEPT_PRECISION = 1e-6  # basically a flat prior on the intercept
ALPHA_0, BETA_0 = 1.0, 1.0


def load():
    d = pd.read_csv(ROOT / "data" / "louisville_merged.csv", parse_dates=["date"])
    d = d[d["has_pollen"] & d["month"].between(3, 6)].copy()
    d["doy_sq"] = d["doy"] ** 2
    return d.dropna(subset=FEATURES + ["log_tree"])


def design(df, mu, sd):
    # standardize with the training mean/sd, then add a column of 1s for the intercept
    z = (df[FEATURES] - mu) / sd
    return np.column_stack([np.ones(len(df)), z.to_numpy()])


class BayesLinReg:
    def fit(self, X, y, prior_precision=PRIOR_PRECISION):
        n, p = X.shape
        Sigma_0_inv = np.diag([INTERCEPT_PRECISION] + [prior_precision] * (p - 1))

        # closed-form posterior (prior on w is scaled by sigma^2, so no Gibbs sampling needed)
        V_n_inv = X.T @ X + Sigma_0_inv
        self.V_n = np.linalg.inv(V_n_inv)
        self.w_n = self.V_n @ X.T @ y
        self.alpha_n = ALPHA_0 + n / 2
        self.beta_n = BETA_0 + 0.5 * (y @ y - self.w_n @ V_n_inv @ self.w_n)
        return self

    def coef_interval(self, level=0.95):
        # each w_j has a Student-t posterior with 2*alpha_n degrees of freedom
        scale = np.sqrt(self.beta_n / self.alpha_n * np.diag(self.V_n))
        t = stats.t.ppf(0.5 + level / 2, df=2 * self.alpha_n)
        return self.w_n - t * scale, self.w_n + t * scale

    def predict(self, X, level=0.9):
        # posterior predictive is also Student-t
        mean = X @ self.w_n
        scale = np.sqrt(self.beta_n / self.alpha_n * (1 + np.sum((X @ self.V_n) * X, axis=1)))
        t = stats.t.ppf(0.5 + level / 2, df=2 * self.alpha_n)
        return mean, mean - t * scale, mean + t * scale


def rmse(y, yhat):
    return float(np.sqrt(np.mean((y - yhat) ** 2)))


def pollen_level(log_counts):
    # CDC tree levels: 0 = Low (<15), 1 = Moderate (15-90), 2 = High (90-1500), 3 = Very High
    g = np.expm1(log_counts) + 1e-9   # small offset so a count of exactly 15 or 90 isn't rounded down
    return np.select([g >= 1500, g >= 90, g >= 15], [3, 2, 1], 0)


def print_metrics(y, preds):
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


def main():
    d = load()
    train, test = d[d["year"] < TEST_START_YEAR], d[d["year"] >= TEST_START_YEAR]
    mu, sd = train[FEATURES].mean(), train[FEATURES].std()
    Xtr, Xte = design(train, mu, sd), design(test, mu, sd)
    ytr, yte = train["log_tree"].to_numpy(), test["log_tree"].to_numpy()

    model = BayesLinReg().fit(Xtr, ytr)
    mean, lo, hi = model.predict(Xte)

    # baselines: day-of-year average and "same as yesterday"
    clim = train.groupby("doy")["log_tree"].mean()
    clim = clim.rolling(7, center=True, min_periods=1).mean()
    clim_pred = test["doy"].map(clim).fillna(ytr.mean()).to_numpy()
    persist = test["log_tree_lag1"].to_numpy()
    lag_only = BayesLinReg().fit(Xtr[:, :2], ytr).predict(Xte[:, :2])[0]

    # same model without any weather features (intercept, yesterday's pollen, day of year, day of year^2)
    no_weather = [0, 1, 2, 3]
    lag_season = BayesLinReg().fit(Xtr[:, no_weather], ytr).predict(Xte[:, no_weather])[0]

    print(f"train days {len(train)}  test days {len(test)}\n")
    print("Test metrics on log(1 + grains/m^3):")
    print_metrics(yte, {"climatology (day of year)": clim_pred,
                        "persistence (yesterday)": persist,
                        "Bayes, lag only": lag_only,
                        "Bayes, lag + season (no weather)": lag_season,
                        "Bayes, lag + season + weather": mean})
    _, lo50, hi50 = model.predict(Xte, level=0.5)
    print(f"\npredictive interval coverage on test: "
          f"50% -> {np.mean((yte >= lo50) & (yte <= hi50)):.1%}, "
          f"90% -> {np.mean((yte >= lo) & (yte <= hi)):.1%}")
    print(f"median 90% interval spans a factor of {np.exp(np.median(hi - lo)):.0f} in pollen count")
    print(f"typical miss (median): factor of {np.exp(np.median(np.abs(yte - mean))):.2f}")
    print(f"posterior mean noise sd: {np.sqrt(model.beta_n / (model.alpha_n - 1)):.3f}\n")

    clo, chi = model.coef_interval()
    names = ["intercept"] + FEATURES
    print(f"{'coefficient (per 1 sd)':24s} {'mean':>7s}   95% posterior interval")
    for n, m, l, h in zip(names, model.w_n, clo, chi):
        flag = "" if l < 0 < h else "  *"
        print(f"  {n:22s} {m:+7.3f}   [{l:+.3f}, {h:+.3f}]{flag}")
    print("  (* = interval excludes 0)")

    # prior sensitivity: refit with weaker and stronger priors
    print("\nprior sensitivity (coefficient prior precision -> test RMSE, 90% coverage):")
    for lam in [0.01, 1, 10, 100, 1000]:
        p, l90, h90 = BayesLinReg().fit(Xtr, ytr, prior_precision=lam).predict(Xte)
        print(f"  {lam:>7g}: RMSE {rmse(yte, p):.3f}, coverage {np.mean((yte >= l90) & (yte <= h90)):.1%}")

    plot(test, yte, mean, lo, hi, names[1:], model.w_n[1:], clo[1:], chi[1:])


def plot(test, yte, mean, lo, hi, names, m, clo, chi):
    FIG.mkdir(exist_ok=True)

    show = test["year"].to_numpy() == 2019   # most complete test season
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
    ax.set_xlabel("effect on log pollen per 1 sd (95% posterior interval)")
    ax.set_title("Posterior coefficients")
    fig.tight_layout()
    fig.savefig(FIG / "coefficients.png", dpi=150)
    print("\nsaved figures/forecast_2019.png and figures/coefficients.png")


if __name__ == "__main__":
    main()
