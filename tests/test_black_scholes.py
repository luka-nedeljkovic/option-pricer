import numpy as np
import pytest

from option_pricer.black_scholes import bs_delta, bs_price

# Reference case used in most textbooks (Hull)
S, K, T, R, SIGMA = 100.0, 100.0, 1.0, 0.05, 0.2


def test_reference_call_price():
    assert bs_price(S, K, T, R, SIGMA, "call") == pytest.approx(10.4506, abs=1e-4)


def test_reference_put_price():
    assert bs_price(S, K, T, R, SIGMA, "put") == pytest.approx(5.5735, abs=1e-4)


@pytest.mark.parametrize("S0", [80.0, 100.0, 120.0])
@pytest.mark.parametrize("T0", [0.25, 1.0, 3.0])
def test_put_call_parity(S0, T0):
    call = bs_price(S0, K, T0, R, SIGMA, "call")
    put = bs_price(S0, K, T0, R, SIGMA, "put")
    assert call - put == pytest.approx(S0 - K * np.exp(-R * T0), abs=1e-10)


def test_deep_in_the_money_call_tends_to_forward_intrinsic():
    call = bs_price(1000.0, K, T, R, SIGMA, "call")
    assert call == pytest.approx(1000.0 - K * np.exp(-R * T), rel=1e-8)


def test_delta_call_minus_put_equals_one():
    diff = bs_delta(S, K, T, R, SIGMA, "call") - bs_delta(S, K, T, R, SIGMA, "put")
    assert diff == pytest.approx(1.0)


def test_delta_matches_finite_difference():
    h = 1e-4
    up = bs_price(S + h, K, T, R, SIGMA)
    down = bs_price(S - h, K, T, R, SIGMA)
    numerical = (up - down) / (2 * h)
    assert bs_delta(S, K, T, R, SIGMA) == pytest.approx(numerical, abs=1e-6)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"T": 0.0},
        {"sigma": 0.0},
        {"S": -1.0},
        {"option_type": "straddle"},
    ],
)
def test_invalid_inputs_raise(kwargs):
    params = {"S": S, "K": K, "T": T, "r": R, "sigma": SIGMA, "option_type": "call"}
    params.update(kwargs)
    with pytest.raises(ValueError):
        bs_price(**params)