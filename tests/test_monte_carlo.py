import pytest

from option_pricer.black_scholes import bs_price
from option_pricer.monte_carlo import mc_price

S, K, T, R, SIGMA = 100.0, 100.0, 1.0, 0.05, 0.2


@pytest.mark.parametrize("option_type", ["call", "put"])
@pytest.mark.parametrize("antithetic", [False, True])
def test_matches_closed_form_within_four_std_errors(option_type, antithetic):
    result = mc_price(
        S, K, T, R, SIGMA, option_type, n_paths=200_000, antithetic=antithetic, seed=0
    )
    exact = bs_price(S, K, T, R, SIGMA, option_type)
    assert abs(result.price - exact) < 4 * result.std_error


def test_confidence_interval_has_95_percent_coverage():
    # A single 95% interval misses the true price 1 time in 20, so we check
    # the coverage frequency over many independent runs instead.
    exact = bs_price(S, K, T, R, SIGMA)
    n_runs = 1_000
    hits = 0
    for seed in range(n_runs):
        low, high = mc_price(
            S, K, T, R, SIGMA, n_paths=5_000, seed=seed
        ).confidence_interval()
        hits += low < exact < high
    assert 0.93 < hits / n_runs < 0.97


def test_antithetic_reduces_std_error_at_equal_cost():
    plain = mc_price(S, K, T, R, SIGMA, n_paths=100_000, seed=2)
    anti = mc_price(S, K, T, R, SIGMA, n_paths=100_000, antithetic=True, seed=2)
    assert anti.std_error < plain.std_error


def test_std_error_scales_like_inverse_sqrt_n():
    small = mc_price(S, K, T, R, SIGMA, n_paths=25_000, seed=3)
    large = mc_price(S, K, T, R, SIGMA, n_paths=400_000, seed=3)
    # 16 times more paths should divide the standard error by about 4.
    assert small.std_error / large.std_error == pytest.approx(4.0, rel=0.05)


def test_same_seed_gives_same_result():
    assert mc_price(S, K, T, R, SIGMA, seed=4) == mc_price(S, K, T, R, SIGMA, seed=4)


@pytest.mark.parametrize(
    "kwargs", [{"n_paths": 1}, {"n_paths": 1001, "antithetic": True}]
)
def test_invalid_path_counts_raise(kwargs):
    with pytest.raises(ValueError):
        mc_price(S, K, T, R, SIGMA, **kwargs)