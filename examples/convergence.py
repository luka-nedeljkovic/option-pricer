"""Convergence of the Monte Carlo pricer to the Black-Scholes closed form.

For each number of paths N, the pricer is run with several independent seeds
and the root-mean-square error against the closed-form price is plotted on a
log-log scale, with and without antithetic variates.

Run from the repository root:
    python examples/convergence.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from option_pricer.black_scholes import bs_price
from option_pricer.monte_carlo import mc_price

S, K, T, R, SIGMA = 100.0, 100.0, 1.0, 0.05, 0.2
N_SEEDS = 200
PATH_COUNTS = np.unique(np.logspace(2, 6, 13).astype(int) // 2 * 2)
OUTPUT = Path(__file__).resolve().parent.parent / "docs" / "mc_convergence.png"


def rmse(n_paths, antithetic):
    exact = bs_price(S, K, T, R, SIGMA)
    errors = [
        mc_price(
            S, K, T, R, SIGMA, n_paths=n_paths, antithetic=antithetic, seed=seed
        ).price
        - exact
        for seed in range(N_SEEDS)
    ]
    return np.sqrt(np.mean(np.square(errors)))


def main():
    plain = np.array([rmse(n, antithetic=False) for n in PATH_COUNTS])
    anti = np.array([rmse(n, antithetic=True) for n in PATH_COUNTS])
    # Reference line of slope -1/2, anchored on the first plain point.
    reference = plain[0] * np.sqrt(PATH_COUNTS[0] / PATH_COUNTS)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.loglog(
        PATH_COUNTS, reference, "--", color="#8a8984", lw=1.5, label=r"slope $-1/2$"
    )
    ax.loglog(
        PATH_COUNTS, plain, "o-", color="#2a78d6", lw=2, ms=6, label="plain Monte Carlo"
    )
    ax.loglog(
        PATH_COUNTS,
        anti,
        "s-",
        color="#eb6834",
        lw=2,
        ms=6,
        label="antithetic variates",
    )

    ax.set_xlabel("number of simulated prices $N$")
    ax.set_ylabel(f"RMSE vs closed form ({N_SEEDS} seeds)")
    ax.set_title(
        "European call, S = K = 100, T = 1, r = 5%, σ = 20%", fontsize=11, loc="left"
    )
    ax.grid(True, which="major", color="#e4e3df", lw=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False)
    fig.tight_layout()

    OUTPUT.parent.mkdir(exist_ok=True)
    fig.savefig(OUTPUT, dpi=150)
    print(f"Saved {OUTPUT}")
    print(f"Variance ratio plain/antithetic: {np.mean((plain / anti) ** 2):.2f}")


if __name__ == "__main__":
    main()