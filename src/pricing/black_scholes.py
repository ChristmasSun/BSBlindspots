import math
from scipy.stats import norm


def bs_d1(S: float, K: float, r: float, sigma: float, T: float) -> float:
    return (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))


def bs_d2(S: float, K: float, r: float, sigma: float, T: float) -> float:
    return bs_d1(S, K, r, sigma, T) - sigma * math.sqrt(T)


def bs_call_price(S: float, K: float, r: float, sigma: float, T: float) -> float:
    if T <= 0:
        return max(S - K, 0.0)
    if sigma <= 0 or S <= 0 or K <= 0:
        return max(S - K, 0.0)
    d1 = bs_d1(S, K, r, sigma, T)
    d2 = bs_d2(S, K, r, sigma, T)
    return S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)


def bs_put_price(S: float, K: float, r: float, sigma: float, T: float) -> float:
    if T <= 0:
        return max(K - S, 0.0)
    if sigma <= 0 or S <= 0 or K <= 0:
        return max(K - S, 0.0)
    d1 = bs_d1(S, K, r, sigma, T)
    d2 = bs_d2(S, K, r, sigma, T)
    return K * math.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


def bs_price(S: float, K: float, r: float, sigma: float, T: float, option_type: str) -> float:
    option_type = option_type.lower()
    if option_type == "call":
        return bs_call_price(S, K, r, sigma, T)
    elif option_type == "put":
        return bs_put_price(S, K, r, sigma, T)
    else:
        raise ValueError(f"option_type must be 'call' or 'put', got '{option_type}'")


def bs_greeks(S: float, K: float, r: float, sigma: float, T: float, option_type: str) -> dict:
    option_type = option_type.lower()
    if option_type not in ("call", "put"):
        raise ValueError(f"option_type must be 'call' or 'put', got '{option_type}'")

    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return {
            "delta": 0.0,
            "gamma": 0.0,
            "theta": 0.0,
            "vega": 0.0,
            "rho": 0.0,
        }

    d1 = bs_d1(S, K, r, sigma, T)
    d2 = bs_d2(S, K, r, sigma, T)
    sqrt_T = math.sqrt(T)
    n_d1 = norm.pdf(d1)
    discount = math.exp(-r * T)

    gamma = n_d1 / (S * sigma * sqrt_T)
    vega = S * n_d1 * sqrt_T / 100.0

    if option_type == "call":
        delta = norm.cdf(d1)
        theta = (
            -S * n_d1 * sigma / (2.0 * sqrt_T)
            - r * K * discount * norm.cdf(d2)
        ) / 365.0
        rho = K * T * discount * norm.cdf(d2) / 100.0
    else:
        delta = norm.cdf(d1) - 1.0
        theta = (
            -S * n_d1 * sigma / (2.0 * sqrt_T)
            + r * K * discount * norm.cdf(-d2)
        ) / 365.0
        rho = -K * T * discount * norm.cdf(-d2) / 100.0

    return {
        "delta": delta,
        "gamma": gamma,
        "theta": theta,
        "vega": vega,
        "rho": rho,
    }


def abs_error(market_price: float, bs_price: float) -> float:
    return abs(market_price - bs_price)


def signed_error(market_price: float, bs_price: float) -> float:
    return market_price - bs_price


def rel_error(market_price: float, bs_price: float) -> float:
    if market_price == 0.0:
        return 0.0
    return (market_price - bs_price) / market_price


def sq_error(market_price: float, bs_price: float) -> float:
    diff = market_price - bs_price
    return diff * diff


if __name__ == "__main__":
    S = 100.0
    K = 100.0
    r = 0.05
    sigma = 0.20
    T = 1.0

    call = bs_call_price(S, K, r, sigma, T)
    put = bs_put_price(S, K, r, sigma, T)
    print(f"S={S}, K={K}, r={r}, sigma={sigma}, T={T}")
    print(f"Call price: {call:.4f}")
    print(f"Put price:  {put:.4f}")

    parity_lhs = call - put
    parity_rhs = S - K * math.exp(-r * T)
    print(f"Put-call parity check: C-P={parity_lhs:.4f}, S-Ke^(-rT)={parity_rhs:.4f}")

    greeks = bs_greeks(S, K, r, sigma, T, "call")
    print(f"Call greeks: {greeks}")

    greeks_put = bs_greeks(S, K, r, sigma, T, "put")
    print(f"Put greeks:  {greeks_put}")

    market = 11.0
    bs = call
    print(f"\nError metrics (market={market}, bs={bs:.4f}):")
    print(f"  abs_error:    {abs_error(market, bs):.4f}")
    print(f"  signed_error: {signed_error(market, bs):.4f}")
    print(f"  rel_error:    {rel_error(market, bs):.4f}")
    print(f"  sq_error:     {sq_error(market, bs):.4f}")
