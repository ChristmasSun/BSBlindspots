import math

import pytest

from src.pricing.black_scholes import (
    abs_error,
    bs_call_price,
    bs_greeks,
    bs_price,
    bs_put_price,
    rel_error,
    signed_error,
    sq_error,
)


# ── Known values ──────────────────────────────────────────────────────


class TestKnownValues:
    def test_atm_call(self) -> None:
        price: float = bs_call_price(S=100, K=100, r=0.05, sigma=0.20, T=1.0)
        assert math.isclose(price, 10.4506, abs_tol=0.001)

    def test_atm_put(self) -> None:
        price: float = bs_put_price(S=100, K=100, r=0.05, sigma=0.20, T=1.0)
        assert math.isclose(price, 5.5735, abs_tol=0.001)

    def test_deep_itm_call(self) -> None:
        price: float = bs_call_price(S=150, K=100, r=0.05, sigma=0.20, T=1.0)
        intrinsic: float = 150.0 - 100.0
        assert price > intrinsic
        assert math.isclose(price, intrinsic, rel_tol=0.10)

    def test_deep_otm_put(self) -> None:
        price: float = bs_put_price(S=150, K=100, r=0.05, sigma=0.20, T=1.0)
        assert price < 0.5

    def test_bs_price_dispatcher(self) -> None:
        call: float = bs_price(S=100, K=100, r=0.05, sigma=0.20, T=1.0, option_type="call")
        put: float = bs_price(S=100, K=100, r=0.05, sigma=0.20, T=1.0, option_type="put")
        assert math.isclose(call, 10.4506, abs_tol=0.001)
        assert math.isclose(put, 5.5735, abs_tol=0.001)

    def test_bs_price_invalid_type(self) -> None:
        with pytest.raises(ValueError):
            bs_price(S=100, K=100, r=0.05, sigma=0.20, T=1.0, option_type="straddle")


# ── Put-call parity ──────────────────────────────────────────────────


class TestPutCallParity:
    def test_parity_atm(self) -> None:
        S: float = 100.0
        K: float = 100.0
        r: float = 0.05
        sigma: float = 0.20
        T: float = 1.0
        call: float = bs_call_price(S, K, r, sigma, T)
        put: float = bs_put_price(S, K, r, sigma, T)
        lhs: float = call - put
        rhs: float = S - K * math.exp(-r * T)
        assert math.isclose(lhs, rhs, abs_tol=1e-8)

    def test_parity_itm(self) -> None:
        S: float = 120.0
        K: float = 100.0
        r: float = 0.03
        sigma: float = 0.30
        T: float = 0.5
        call: float = bs_call_price(S, K, r, sigma, T)
        put: float = bs_put_price(S, K, r, sigma, T)
        lhs: float = call - put
        rhs: float = S - K * math.exp(-r * T)
        assert math.isclose(lhs, rhs, abs_tol=1e-8)

    def test_parity_otm(self) -> None:
        S: float = 80.0
        K: float = 100.0
        r: float = 0.08
        sigma: float = 0.25
        T: float = 2.0
        call: float = bs_call_price(S, K, r, sigma, T)
        put: float = bs_put_price(S, K, r, sigma, T)
        lhs: float = call - put
        rhs: float = S - K * math.exp(-r * T)
        assert math.isclose(lhs, rhs, abs_tol=1e-8)

    def test_parity_short_dated(self) -> None:
        S: float = 100.0
        K: float = 105.0
        r: float = 0.02
        sigma: float = 0.15
        T: float = 0.05
        call: float = bs_call_price(S, K, r, sigma, T)
        put: float = bs_put_price(S, K, r, sigma, T)
        lhs: float = call - put
        rhs: float = S - K * math.exp(-r * T)
        assert math.isclose(lhs, rhs, abs_tol=1e-8)


# ── Edge cases ────────────────────────────────────────────────────────


class TestEdgeCases:
    def test_zero_time_call_itm(self) -> None:
        price: float = bs_call_price(S=110, K=100, r=0.05, sigma=0.20, T=0.0)
        assert math.isclose(price, 10.0, abs_tol=1e-10)

    def test_zero_time_call_otm(self) -> None:
        price: float = bs_call_price(S=90, K=100, r=0.05, sigma=0.20, T=0.0)
        assert math.isclose(price, 0.0, abs_tol=1e-10)

    def test_zero_time_put_itm(self) -> None:
        price: float = bs_put_price(S=90, K=100, r=0.05, sigma=0.20, T=0.0)
        assert math.isclose(price, 10.0, abs_tol=1e-10)

    def test_zero_time_put_otm(self) -> None:
        price: float = bs_put_price(S=110, K=100, r=0.05, sigma=0.20, T=0.0)
        assert math.isclose(price, 0.0, abs_tol=1e-10)

    def test_zero_sigma_call_itm(self) -> None:
        price: float = bs_call_price(S=110, K=100, r=0.05, sigma=0.0, T=1.0)
        assert math.isclose(price, 10.0, abs_tol=1e-10)

    def test_zero_sigma_put_itm(self) -> None:
        price: float = bs_put_price(S=90, K=100, r=0.05, sigma=0.0, T=1.0)
        assert math.isclose(price, 10.0, abs_tol=1e-10)

    def test_zero_spot_call(self) -> None:
        price: float = bs_call_price(S=0, K=100, r=0.05, sigma=0.20, T=1.0)
        assert math.isclose(price, 0.0, abs_tol=1e-10)

    def test_zero_strike_call(self) -> None:
        price: float = bs_call_price(S=100, K=0, r=0.05, sigma=0.20, T=1.0)
        assert math.isclose(price, 100.0, abs_tol=1e-10)


# ── Greeks sanity checks ─────────────────────────────────────────────


class TestGreeks:
    def test_atm_call_delta(self) -> None:
        greeks: dict = bs_greeks(S=100, K=100, r=0.05, sigma=0.20, T=1.0, option_type="call")
        assert 0.5 <= greeks["delta"] <= 0.7

    def test_atm_put_delta(self) -> None:
        greeks: dict = bs_greeks(S=100, K=100, r=0.05, sigma=0.20, T=1.0, option_type="put")
        assert -0.3 >= greeks["delta"] >= -0.5

    def test_call_gamma_positive(self) -> None:
        greeks: dict = bs_greeks(S=100, K=100, r=0.05, sigma=0.20, T=1.0, option_type="call")
        assert greeks["gamma"] > 0

    def test_put_gamma_positive(self) -> None:
        greeks: dict = bs_greeks(S=100, K=100, r=0.05, sigma=0.20, T=1.0, option_type="put")
        assert greeks["gamma"] > 0

    def test_call_vega_positive(self) -> None:
        greeks: dict = bs_greeks(S=100, K=100, r=0.05, sigma=0.20, T=1.0, option_type="call")
        assert greeks["vega"] > 0

    def test_put_vega_positive(self) -> None:
        greeks: dict = bs_greeks(S=100, K=100, r=0.05, sigma=0.20, T=1.0, option_type="put")
        assert greeks["vega"] > 0

    def test_call_put_delta_sum(self) -> None:
        call_greeks: dict = bs_greeks(S=100, K=100, r=0.05, sigma=0.20, T=1.0, option_type="call")
        put_greeks: dict = bs_greeks(S=100, K=100, r=0.05, sigma=0.20, T=1.0, option_type="put")
        delta_sum: float = call_greeks["delta"] + abs(put_greeks["delta"])
        assert math.isclose(delta_sum, 1.0, abs_tol=1e-8)

    def test_gamma_same_for_call_and_put(self) -> None:
        call_greeks: dict = bs_greeks(S=100, K=100, r=0.05, sigma=0.20, T=1.0, option_type="call")
        put_greeks: dict = bs_greeks(S=100, K=100, r=0.05, sigma=0.20, T=1.0, option_type="put")
        assert math.isclose(call_greeks["gamma"], put_greeks["gamma"], abs_tol=1e-10)

    def test_vega_same_for_call_and_put(self) -> None:
        call_greeks: dict = bs_greeks(S=100, K=100, r=0.05, sigma=0.20, T=1.0, option_type="call")
        put_greeks: dict = bs_greeks(S=100, K=100, r=0.05, sigma=0.20, T=1.0, option_type="put")
        assert math.isclose(call_greeks["vega"], put_greeks["vega"], abs_tol=1e-10)

    def test_greeks_invalid_type(self) -> None:
        with pytest.raises(ValueError):
            bs_greeks(S=100, K=100, r=0.05, sigma=0.20, T=1.0, option_type="straddle")


# ── Error metrics ─────────────────────────────────────────────────────


class TestErrorMetrics:
    def test_abs_error(self) -> None:
        assert math.isclose(abs_error(10.0, 8.0), 2.0, abs_tol=1e-10)

    def test_signed_error_underpriced(self) -> None:
        result: float = signed_error(10.0, 8.0)
        assert math.isclose(result, 2.0, abs_tol=1e-10)

    def test_signed_error_overpriced(self) -> None:
        result: float = signed_error(8.0, 10.0)
        assert math.isclose(result, -2.0, abs_tol=1e-10)

    def test_rel_error(self) -> None:
        result: float = rel_error(10.0, 8.0)
        assert math.isclose(result, 0.2, abs_tol=1e-10)

    def test_rel_error_zero_market(self) -> None:
        result: float = rel_error(0.0, 5.0)
        assert math.isclose(result, 0.0, abs_tol=1e-10)

    def test_sq_error(self) -> None:
        result: float = sq_error(10.0, 8.0)
        assert math.isclose(result, 4.0, abs_tol=1e-10)


# ── Monotonicity ──────────────────────────────────────────────────────


class TestMonotonicity:
    def test_call_increases_with_spot(self) -> None:
        spot_values: list[float] = [80.0, 90.0, 100.0, 110.0, 120.0]
        prices: list[float] = []
        for s in spot_values:
            prices.append(bs_call_price(S=s, K=100, r=0.05, sigma=0.20, T=1.0))
        for i in range(len(prices) - 1):
            assert prices[i] < prices[i + 1]

    def test_put_increases_with_strike(self) -> None:
        strike_values: list[float] = [80.0, 90.0, 100.0, 110.0, 120.0]
        prices: list[float] = []
        for k in strike_values:
            prices.append(bs_put_price(S=100, K=k, r=0.05, sigma=0.20, T=1.0))
        for i in range(len(prices) - 1):
            assert prices[i] < prices[i + 1]

    def test_call_increases_with_sigma(self) -> None:
        sigma_values: list[float] = [0.10, 0.20, 0.30, 0.40, 0.50]
        prices: list[float] = []
        for sig in sigma_values:
            prices.append(bs_call_price(S=100, K=100, r=0.05, sigma=sig, T=1.0))
        for i in range(len(prices) - 1):
            assert prices[i] < prices[i + 1]

    def test_put_increases_with_sigma(self) -> None:
        sigma_values: list[float] = [0.10, 0.20, 0.30, 0.40, 0.50]
        prices: list[float] = []
        for sig in sigma_values:
            prices.append(bs_put_price(S=100, K=100, r=0.05, sigma=sig, T=1.0))
        for i in range(len(prices) - 1):
            assert prices[i] < prices[i + 1]

    def test_call_increases_with_time(self) -> None:
        time_values: list[float] = [0.1, 0.25, 0.5, 1.0, 2.0]
        prices: list[float] = []
        for t in time_values:
            prices.append(bs_call_price(S=100, K=100, r=0.05, sigma=0.20, T=t))
        for i in range(len(prices) - 1):
            assert prices[i] < prices[i + 1]

    def test_put_increases_with_time(self) -> None:
        time_values: list[float] = [0.1, 0.25, 0.5, 1.0, 2.0]
        prices: list[float] = []
        for t in time_values:
            prices.append(bs_put_price(S=100, K=100, r=0.05, sigma=0.20, T=t))
        for i in range(len(prices) - 1):
            assert prices[i] < prices[i + 1]
