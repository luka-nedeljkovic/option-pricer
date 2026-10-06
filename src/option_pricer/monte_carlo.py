"""Monte Carlo pricing of European options under Black-Scholes.

The terminal price is sampled exactly from its lognormal law under the
risk-neutral measure, so there is no time-discretisation error: the only
error is statistical and is reported through a standard error.
"""

from dataclasses import dataclass

import numpy as np

from option_pricer.black_scholes import _check_inputs


@dataclass(frozen=True)
class MCResult:
    """Monte Carlo estimate with its standard error."""

    price: float
    std_error: float

    def confidence_interval(self, z=1.96):
        """Asymptotic confidence interval (95% by default)."""
        return self.price - z * self.std_error, self.price + z * self.std_error


def mc_price(
    S,
    K,
    T,
    r,
    sigma,
    option_type="call",
    n_paths=100_000,
    antithetic=False,
    seed=None,
):
    """Monte Carlo price of a European call or put under Black-Scholes.

    With ``antithetic=True``, half of the normal draws are reused with the
    opposite sign, so the total number of simulated prices is still
    ``n_paths`` and the comparison with plain Monte Carlo is at equal cost.
    """
    _check_inputs(S, K, T, sigma, option_type)
    if n_paths < 2:
        raise ValueError("n_paths must be at least 2")
    if antithetic and n_paths % 2 != 0:
        raise ValueError("n_paths must be even when antithetic=True")

    rng = np.random.default_rng(seed)
    discount = np.exp(-r * T)
    drift = (r - 0.5 * sigma**2) * T
    vol = sigma * np.sqrt(T)

    def discounted_payoff(z):
        s_T = S * np.exp(drift + vol * z)
        if option_type == "call":
            return discount * np.maximum(s_T - K, 0.0)
        return discount * np.maximum(K - s_T, 0.0)

    if antithetic:
        z = rng.standard_normal(n_paths // 2)
        # Average each pair first: the pairs are i.i.d., the paths are not.
        samples = 0.5 * (discounted_payoff(z) + discounted_payoff(-z))
    else:
        samples = discounted_payoff(rng.standard_normal(n_paths))

    price = samples.mean()
    std_error = samples.std(ddof=1) / np.sqrt(samples.size)
    return MCResult(float(price), float(std_error))