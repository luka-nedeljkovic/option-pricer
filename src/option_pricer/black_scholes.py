"""Black-Scholes closed-form prices and delta for European options.

Conventions: T in years, r and sigma annualised, continuous compounding,
no dividends.
"""

import numpy as np
from scipy.stats import norm


def _check_inputs(S, K, T, sigma, option_type):
    if S <= 0 or K <= 0:
        raise ValueError("S and K must be positive")
    if T <= 0:
        raise ValueError("T must be positive")
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    if option_type not in ("call", "put"):
        raise ValueError("option_type must be 'call' or 'put'")


def _d1_d2(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return d1, d2


def bs_price(S, K, T, r, sigma, option_type="call"):
    """Price of a European call or put under Black-Scholes."""
    _check_inputs(S, K, T, sigma, option_type)
    d1, d2 = _d1_d2(S, K, T, r, sigma)
    discount = np.exp(-r * T)
    if option_type == "call":
        return S * norm.cdf(d1) - K * discount * norm.cdf(d2)
    return K * discount * norm.cdf(-d2) - S * norm.cdf(-d1)


def bs_delta(S, K, T, r, sigma, option_type="call"):
    """Delta (dV/dS) of a European call or put under Black-Scholes."""
    _check_inputs(S, K, T, sigma, option_type)
    d1, _ = _d1_d2(S, K, T, r, sigma)
    if option_type == "call":
        return norm.cdf(d1)
    return norm.cdf(d1) - 1.0