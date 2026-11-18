"""Second method: a multilayer perceptron (MLP) for daily tree pollen, in NumPy.

Same target, features and train/test split as scripts/bayes_regression.py, so the
two models are directly comparable:
  target  log(1 + tree pollen grains/m^3), March-June
  train   2003-2014 seasons (fit weights)
  val     2015-2017 seasons (early stopping only)
  test    2018-2022 seasons (reported once at the end)

Network: inputs -> 32 ReLU -> 16 ReLU -> 1 linear, trained with Adam on mean
squared error plus L2 weight decay. Results are averaged over several random
seeds because small networks vary run to run.

Run from the repo root:  .venv/bin/python scripts/mlp.py
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from bayes_regression import (FEATURES, TEST_START_YEAR, BayesLinReg, design,
                              load, rmse)

ROOT = Path(__file__).resolve().parents[1]
VAL_START_YEAR = 2015
HIDDEN = (32, 16)
LR, WEIGHT_DECAY = 1e-3, 1e-4
BATCH, MAX_EPOCHS, PATIENCE = 64, 500, 30
SEEDS = range(5)


class MLP:
    def __init__(self, n_in: int, hidden, rng: np.random.Generator):
        sizes = [n_in, *hidden, 1]
        # He initialization suits ReLU layers.
        self.W = [rng.normal(0, np.sqrt(2 / a), (a, b)) for a, b in zip(sizes, sizes[1:])]
        self.b = [np.zeros(b) for b in sizes[1:]]

    def forward(self, X):
        acts = [X]
        for i, (W, b) in enumerate(zip(self.W, self.b)):
            z = acts[-1] @ W + b
            acts.append(z if i == len(self.W) - 1 else np.maximum(z, 0))
        return acts

    def predict(self, X):
        return self.forward(X)[-1][:, 0]

    def grads(self, X, y):
        acts = self.forward(X)
        delta = 2 * (acts[-1][:, 0] - y)[:, None] / len(y)  # d(MSE)/d(output)
        gW, gb = [], []
        for i in reversed(range(len(self.W))):
            gW.insert(0, acts[i].T @ delta + WEIGHT_DECAY * self.W[i])
            gb.insert(0, delta.sum(0))
            if i > 0:
                delta = (delta @ self.W[i].T) * (acts[i] > 0)  # back through ReLU
        return gW, gb

    @property
    def params(self):
        return self.W + self.b


def train(Xtr, ytr, Xval, yval, seed: int) -> tuple[MLP, list, list]:
    rng = np.random.default_rng(seed)
    net = MLP(Xtr.shape[1], HIDDEN, rng)
    m = [np.zeros_like(p) for p in net.params]
    v = [np.zeros_like(p) for p in net.params]
    b1, b2, eps, step = 0.9, 0.999, 1e-8, 0
    best, best_params, wait = np.inf, None, 0
    hist_tr, hist_val = [], []

    for epoch in range(MAX_EPOCHS):
        order = rng.permutation(len(ytr))
        for start in range(0, len(order), BATCH):
            idx = order[start:start + BATCH]
            gW, gb = net.grads(Xtr[idx], ytr[idx])
            step += 1
            for p, g, mi, vi in zip(net.params, gW + gb, m, v):  # Adam update
                mi[:] = b1 * mi + (1 - b1) * g
                vi[:] = b2 * vi + (1 - b2) * g * g
                p -= LR * (mi / (1 - b1 ** step)) / (np.sqrt(vi / (1 - b2 ** step)) + eps)

        hist_tr.append(rmse(ytr, net.predict(Xtr)))
        hist_val.append(rmse(yval, net.predict(Xval)))
        if hist_val[-1] < best - 1e-4:
            best, wait = hist_val[-1], 0
            best_params = [p.copy() for p in net.params]
        else:
            wait += 1
            if wait >= PATIENCE:
                break

    for p, saved in zip(net.params, best_params):  # restore best-on-validation weights
        p[:] = saved
    return net, hist_tr, hist_val


def main() -> None:
    d = load()
    fit = d[d["year"] < VAL_START_YEAR]
    val = d[(d["year"] >= VAL_START_YEAR) & (d["year"] < TEST_START_YEAR)]
    test = d[d["year"] >= TEST_START_YEAR]

    # Standardize with the fitting years only; drop the intercept column (MLP has biases).
    mu, sd = fit[FEATURES].mean(), fit[FEATURES].std()
    Xfit, Xval, Xte = (design(x, mu, sd)[:, 1:] for x in (fit, val, test))
    yfit, yval, yte = (x["log_tree"].to_numpy() for x in (fit, val, test))

    preds, scores, curves = [], [], None
    for seed in SEEDS:
        net, htr, hval = train(Xfit, yfit, Xval, yval, seed)
        p = net.predict(Xte)
        preds.append(p)
        scores.append(rmse(yte, p))
        print(f"seed {seed}: stopped after {len(hval)} epochs, "
              f"val RMSE {min(hval):.3f}, test RMSE {scores[-1]:.3f}")
        curves = curves or (htr, hval)
    ens = np.mean(preds, axis=0)

    # Bayes model refit on train+val (2003-2017), exactly as in bayes_regression.py.
    trval = d[d["year"] < TEST_START_YEAR]
    mu2, sd2 = trval[FEATURES].mean(), trval[FEATURES].std()
    bayes = BayesLinReg().fit(design(trval, mu2, sd2), trval["log_tree"].to_numpy())
    bmean = bayes.predict(design(test, mu2, sd2))[0]

    print(f"\nTest RMSE on log(1 + grains/m^3), {len(test)} days, 2018-2022:")
    print(f"  Bayesian linear regression     {rmse(yte, bmean):.3f}")
    print(f"  MLP, single seed (mean ± sd)   {np.mean(scores):.3f} ± {np.std(scores):.3f}")
    print(f"  MLP, average of {len(SEEDS)} seeds        {rmse(yte, ens):.3f}")

    plot(test, yte, bmean, ens, curves)


def plot(test, yte, bmean, ens, curves) -> None:
    fig_dir = ROOT / "figures"
    show = test["year"].to_numpy() == 2019
    dates = test["date"].to_numpy()[show]
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(dates, np.expm1(yte[show]), "k.", ms=4, label="observed")
    ax.plot(dates, np.expm1(bmean[show]).clip(0), color="tab:blue", label="Bayesian regression")
    ax.plot(dates, np.expm1(ens[show]).clip(0), color="tab:orange", label="MLP (5-seed average)")
    ax.set_yscale("symlog", linthresh=10)
    ax.set_ylabel("tree pollen (grains/m³)")
    ax.set_title("Louisville tree pollen, 2019 test season: Bayes vs MLP")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(fig_dir / "mlp_vs_bayes_2019.png", dpi=150)

    htr, hval = curves
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.plot(htr, label="train (2003-2014)")
    ax.plot(hval, label="validation (2015-2017)")
    ax.set_xlabel("epoch")
    ax.set_ylabel("RMSE of log(1 + pollen)")
    ax.set_title("MLP training curve (seed 0)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_dir / "mlp_training_curve.png", dpi=150)
    print("\nsaved figures/mlp_vs_bayes_2019.png and figures/mlp_training_curve.png")


if __name__ == "__main__":
    main()
