import math
from pathlib import Path

import pandas as pd
from scipy.stats import norm

from src.pricing.black_scholes import bs_price, bs_d1


def implied_vol(
    market_price: float,
    S: float,
    K: float,
    r: float,
    T: float,
    option_type: str,
    tol: float = 1e-6,
    max_iter: int = 100,
    initial_guess: float = 0.3,
) -> float | None:
    option_type = option_type.lower()

    if T <= 0 or S <= 0 or K <= 0 or market_price <= 0:
        return None

    intrinsic_call = max(S - K * math.exp(-r * T), 0.0)
    intrinsic_put = max(K * math.exp(-r * T) - S, 0.0)

    if option_type == "call" and market_price < intrinsic_call:
        return None
    if option_type == "put" and market_price < intrinsic_put:
        return None

    sigma = initial_guess

    for _ in range(max_iter):
        if sigma < 0.001 or sigma > 10.0:
            return None

        price = bs_price(S, K, r, sigma, T, option_type)
        diff = price - market_price

        if abs(diff) < tol:
            return sigma

        d1 = bs_d1(S, K, r, sigma, T)
        vega = S * norm.pdf(d1) * math.sqrt(T)

        if abs(vega) < 1e-12:
            return None

        sigma = sigma - diff / vega

    return None


def compute_iv_for_chain(options_df: pd.DataFrame, risk_free_rate: float) -> pd.Series:
    iv_values: list[float | None] = []

    for idx in range(len(options_df)):
        row = options_df.iloc[idx]
        iv = implied_vol(
            market_price=row["mid_price"],
            S=row["underlying_price"],
            K=row["strike"],
            r=risk_free_rate,
            T=row["time_to_maturity"],
            option_type=row["option_type"],
        )
        iv_values.append(iv)

    return pd.Series(iv_values, index=options_df.index, name="implied_vol")


def iv_smile(
    options_df: pd.DataFrame,
    risk_free_rate: float,
    expiration: str | None = None,
) -> pd.DataFrame:
    df = options_df.copy()

    if expiration is not None:
        df = df[df["expiration"] == expiration].copy()

    iv_series = compute_iv_for_chain(df, risk_free_rate)
    df["iv"] = iv_series.values

    df = df[df["iv"].notna()].copy()

    strikes: list[float] = []
    moneyness_vals: list[float] = []
    ivs: list[float] = []
    option_types: list[str] = []

    for idx in range(len(df)):
        row = df.iloc[idx]
        strikes.append(row["strike"])
        moneyness_vals.append(row["underlying_price"] / row["strike"])
        ivs.append(row["iv"])
        option_types.append(row["option_type"])

    result = pd.DataFrame({
        "strike": strikes,
        "moneyness": moneyness_vals,
        "iv": ivs,
        "option_type": option_types,
    })

    result = result.sort_values("strike").reset_index(drop=True)

    return result


if __name__ == "__main__":
    data_path = Path(__file__).resolve().parents[2] / "data" / "raw" / "options" / "SPY_current.csv"

    if not data_path.exists():
        print(f"No cached options data found at {data_path}")
        print("Run src/data/fetch_options.py first.")
    else:
        df = pd.read_csv(data_path)
        print(f"Loaded {len(df)} options from {data_path}")

        df["expiration"] = pd.to_datetime(df["expiration"])
        df["date"] = pd.to_datetime(df["date"])

        if "time_to_maturity" not in df.columns:
            dte_days = (df["expiration"] - df["date"]).dt.days
            df["time_to_maturity"] = dte_days / 365.0

        df = df[df["mid_price"] > 0.10].copy()
        df = df[df["time_to_maturity"] > 0].copy()
        print(f"After filters: {len(df)} options")

        r = 0.04

        print("\nComputing implied volatilities...")
        df["implied_vol"] = compute_iv_for_chain(df, r).values

        total = len(df)
        converged = df["implied_vol"].notna().sum()
        failed = total - converged
        print(f"Converged: {converged}/{total} ({100.0 * converged / total:.1f}%)")
        print(f"Failed:    {failed}/{total}")

        valid = df[df["implied_vol"].notna()]
        if len(valid) > 0:
            print(f"\nIV Summary Stats:")
            print(f"  Mean:   {valid['implied_vol'].mean():.4f}")
            print(f"  Median: {valid['implied_vol'].median():.4f}")
            print(f"  Std:    {valid['implied_vol'].std():.4f}")
            print(f"  Min:    {valid['implied_vol'].min():.4f}")
            print(f"  Max:    {valid['implied_vol'].max():.4f}")

            for otype in ["call", "put"]:
                subset = valid[valid["option_type"] == otype]
                if len(subset) > 0:
                    print(f"\n  {otype.upper()}s: n={len(subset)}, "
                          f"mean IV={subset['implied_vol'].mean():.4f}, "
                          f"median IV={subset['implied_vol'].median():.4f}")

        expirations = sorted(df["expiration"].unique())
        if len(expirations) > 0:
            nearest_exp = str(expirations[0])[:10]
            print(f"\nIV smile for nearest expiration ({nearest_exp}):")
            smile = iv_smile(df, r, expiration=expirations[0])
            if len(smile) > 0:
                print(f"  {len(smile)} strikes with valid IV")
                print(f"  Moneyness range: {smile['moneyness'].min():.3f} - {smile['moneyness'].max():.3f}")
                print(f"  IV range: {smile['iv'].min():.4f} - {smile['iv'].max():.4f}")
                print(smile.to_string(index=False))
