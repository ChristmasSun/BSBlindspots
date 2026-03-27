import matplotlib
matplotlib.use("Agg")

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.style.use("seaborn-v0_8-whitegrid")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PLOTS_DIR = PROJECT_ROOT / "data" / "features" / "plots"

VIX_REGIME_LABELS = {
    0: "Low (<15)",
    1: "Medium (15-25)",
    2: "High (>25)",
}


def _save_or_show(fig: plt.Figure, save_path: Optional[str | Path]) -> None:
    if save_path is not None:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_vol_smile(
    strikes: list[float] | np.ndarray,
    ivs: list[float] | np.ndarray,
    option_types: list[str] | np.ndarray,
    expiration_label: Optional[str] = None,
    save_path: Optional[str | Path] = None,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))

    call_strikes: list[float] = []
    call_ivs: list[float] = []
    put_strikes: list[float] = []
    put_ivs: list[float] = []

    for i in range(len(strikes)):
        if option_types[i] == "call":
            call_strikes.append(strikes[i])
            call_ivs.append(ivs[i])
        else:
            put_strikes.append(strikes[i])
            put_ivs.append(ivs[i])

    ax.scatter(call_strikes, call_ivs, c="#1f77b4", alpha=0.6, s=20, label="Call")
    ax.scatter(put_strikes, put_ivs, c="#d62728", alpha=0.6, s=20, label="Put")

    title = "Volatility Smile"
    if expiration_label is not None:
        title = f"Volatility Smile — {expiration_label}"
    ax.set_title(title, fontsize=14)
    ax.set_xlabel("Strike", fontsize=12)
    ax.set_ylabel("Implied Volatility", fontsize=12)
    ax.legend()

    _save_or_show(fig, save_path)


def plot_error_by_moneyness(
    moneyness: np.ndarray | pd.Series,
    errors: np.ndarray | pd.Series,
    error_type: str = "rel_error",
    save_path: Optional[str | Path] = None,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.scatter(moneyness, errors, alpha=0.15, s=8, c="#1f77b4", label="Individual")

    sort_idx = np.argsort(moneyness)
    sorted_m = np.array(moneyness)[sort_idx]
    sorted_e = np.array(errors)[sort_idx]

    window = max(len(sorted_m) // 50, 10)
    smoothed: list[float] = []
    smoothed_x: list[float] = []
    for i in range(window, len(sorted_m) - window):
        smoothed_x.append(sorted_m[i])
        smoothed.append(float(np.mean(sorted_e[i - window : i + window])))

    ax.plot(smoothed_x, smoothed, c="#d62728", linewidth=2, label="Rolling Mean")

    ax.set_title(f"BS Error vs Moneyness ({error_type})", fontsize=14)
    ax.set_xlabel("Moneyness (S/K)", fontsize=12)
    ax.set_ylabel(error_type, fontsize=12)
    ax.legend()

    _save_or_show(fig, save_path)


def plot_error_by_dte(
    dte: np.ndarray | pd.Series,
    errors: np.ndarray | pd.Series,
    error_type: str = "rel_error",
    save_path: Optional[str | Path] = None,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.scatter(dte, errors, alpha=0.15, s=8, c="#1f77b4", label="Individual")

    sort_idx = np.argsort(dte)
    sorted_d = np.array(dte)[sort_idx]
    sorted_e = np.array(errors)[sort_idx]

    window = max(len(sorted_d) // 50, 10)
    smoothed: list[float] = []
    smoothed_x: list[float] = []
    for i in range(window, len(sorted_d) - window):
        smoothed_x.append(sorted_d[i])
        smoothed.append(float(np.mean(sorted_e[i - window : i + window])))

    ax.plot(smoothed_x, smoothed, c="#d62728", linewidth=2, label="Rolling Mean")

    ax.set_title(f"BS Error vs Days to Expiry ({error_type})", fontsize=14)
    ax.set_xlabel("Days to Expiry", fontsize=12)
    ax.set_ylabel(error_type, fontsize=12)
    ax.legend()

    _save_or_show(fig, save_path)


def plot_error_heatmap(
    moneyness: np.ndarray | pd.Series,
    dte: np.ndarray | pd.Series,
    errors: np.ndarray | pd.Series,
    save_path: Optional[str | Path] = None,
) -> None:
    fig, ax = plt.subplots(figsize=(12, 8))

    m_arr = np.array(moneyness)
    d_arr = np.array(dte)
    e_arr = np.array(errors)

    m_bins = np.linspace(m_arr.min(), m_arr.max(), 21)
    d_bins = np.linspace(d_arr.min(), d_arr.max(), 21)

    m_idx = np.digitize(m_arr, m_bins) - 1
    d_idx = np.digitize(d_arr, d_bins) - 1
    m_idx = np.clip(m_idx, 0, len(m_bins) - 2)
    d_idx = np.clip(d_idx, 0, len(d_bins) - 2)

    grid = np.full((len(d_bins) - 1, len(m_bins) - 1), np.nan)
    counts = np.zeros_like(grid)

    for i in range(len(e_arr)):
        row = d_idx[i]
        col = m_idx[i]
        if np.isnan(grid[row, col]):
            grid[row, col] = 0.0
        grid[row, col] += e_arr[i]
        counts[row, col] += 1

    mask = counts > 0
    grid[mask] = grid[mask] / counts[mask]

    m_centers: list[float] = []
    for i in range(len(m_bins) - 1):
        m_centers.append((m_bins[i] + m_bins[i + 1]) / 2)

    d_centers: list[float] = []
    for i in range(len(d_bins) - 1):
        d_centers.append((d_bins[i] + d_bins[i + 1]) / 2)

    im = ax.imshow(
        grid,
        aspect="auto",
        origin="lower",
        cmap="RdBu_r",
        extent=[m_bins[0], m_bins[-1], d_bins[0], d_bins[-1]],
    )
    fig.colorbar(im, ax=ax, label="Mean Error")

    ax.set_title("BS Error Heatmap: Moneyness vs DTE", fontsize=14)
    ax.set_xlabel("Moneyness (S/K)", fontsize=12)
    ax.set_ylabel("Days to Expiry", fontsize=12)

    _save_or_show(fig, save_path)


def plot_regime_comparison(
    df: pd.DataFrame,
    error_col: str = "rel_error",
    regime_col: str = "vix_regime",
    save_path: Optional[str | Path] = None,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))

    filtered = df.dropna(subset=[regime_col, error_col])

    regime_values = sorted(filtered[regime_col].unique())
    data_groups: list[np.ndarray] = []
    labels: list[str] = []
    for rv in regime_values:
        mask = filtered[regime_col] == rv
        data_groups.append(filtered.loc[mask, error_col].values)
        label = VIX_REGIME_LABELS.get(int(rv), str(rv))
        labels.append(label)

    parts = ax.violinplot(data_groups, showmedians=True, showextrema=False)

    colors = ["#2ca02c", "#ff7f0e", "#d62728"]
    for i, pc in enumerate(parts["bodies"]):
        color_idx = min(i, len(colors) - 1)
        pc.set_facecolor(colors[color_idx])
        pc.set_alpha(0.7)

    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels)
    ax.set_title(f"BS Error by VIX Regime ({error_col})", fontsize=14)
    ax.set_xlabel("VIX Regime", fontsize=12)
    ax.set_ylabel(error_col, fontsize=12)

    _save_or_show(fig, save_path)


def plot_earnings_effect(
    df: pd.DataFrame,
    save_path: Optional[str | Path] = None,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))

    filtered = df.dropna(subset=["days_to_earnings", "abs_error"])
    filtered = filtered[filtered["days_to_earnings"] >= 0]

    max_days = 60
    filtered = filtered[filtered["days_to_earnings"] <= max_days]

    day_vals = sorted(filtered["days_to_earnings"].unique())
    mean_errors: list[float] = []
    day_list: list[float] = []
    for d in day_vals:
        mask = filtered["days_to_earnings"] == d
        subset = filtered.loc[mask, "abs_error"]
        if len(subset) >= 3:
            mean_errors.append(float(subset.mean()))
            day_list.append(float(d))

    ax.bar(day_list, mean_errors, width=0.8, color="#1f77b4", alpha=0.7)

    ax.set_title("Mean Absolute BS Error vs Days to Earnings", fontsize=14)
    ax.set_xlabel("Days to Next Earnings", fontsize=12)
    ax.set_ylabel("Mean |Error|", fontsize=12)

    _save_or_show(fig, save_path)


def plot_model_predictions(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray | pd.Series,
    model_name: str = "XGBoost",
    save_path: Optional[str | Path] = None,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 8))

    ax.scatter(y_true, y_pred, alpha=0.15, s=8, c="#1f77b4")

    all_vals = np.concatenate([np.array(y_true), np.array(y_pred)])
    lo = float(np.min(all_vals))
    hi = float(np.max(all_vals))
    margin = (hi - lo) * 0.05
    ax.plot([lo - margin, hi + margin], [lo - margin, hi + margin], "k--", linewidth=1, alpha=0.7)
    ax.set_xlim(lo - margin, hi + margin)
    ax.set_ylim(lo - margin, hi + margin)

    ss_res = float(np.sum((np.array(y_true) - np.array(y_pred)) ** 2))
    ss_tot = float(np.sum((np.array(y_true) - np.mean(y_true)) ** 2))
    r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    ax.text(
        0.05,
        0.95,
        f"R² = {r2:.4f}",
        transform=ax.transAxes,
        fontsize=13,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
    )

    ax.set_title(f"{model_name}: Predicted vs Actual", fontsize=14)
    ax.set_xlabel("Actual", fontsize=12)
    ax.set_ylabel("Predicted", fontsize=12)
    ax.set_aspect("equal")

    _save_or_show(fig, save_path)


if __name__ == "__main__":
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    features_dir = PROJECT_ROOT / "data" / "features"
    csv_path = features_dir / "SPY_features.csv"

    if not csv_path.exists():
        print(f"No cached features at {csv_path}. Run build_features.py first.")
        raise SystemExit(1)

    print(f"Loading {csv_path}...")
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} rows")

    print("Generating vol smile plot...")
    plot_vol_smile(
        strikes=df["strike"].values,
        ivs=df["hist_vol_20"].values,
        option_types=df["option_type"].values,
        expiration_label="SPY (hist vol proxy)",
        save_path=PLOTS_DIR / "vol_smile.png",
    )

    print("Generating error by moneyness plot...")
    plot_error_by_moneyness(
        moneyness=df["moneyness"].values,
        errors=df["rel_error"].values,
        error_type="rel_error",
        save_path=PLOTS_DIR / "error_by_moneyness.png",
    )

    print("Generating error by DTE plot...")
    plot_error_by_dte(
        dte=df["dte"].values,
        errors=df["rel_error"].values,
        error_type="rel_error",
        save_path=PLOTS_DIR / "error_by_dte.png",
    )

    print("Generating error heatmap...")
    plot_error_heatmap(
        moneyness=df["moneyness"].values,
        dte=df["dte"].values,
        errors=df["rel_error"].values,
        save_path=PLOTS_DIR / "error_heatmap.png",
    )

    print("Generating regime comparison plot...")
    plot_regime_comparison(
        df=df,
        error_col="rel_error",
        regime_col="vix_regime",
        save_path=PLOTS_DIR / "regime_comparison.png",
    )

    print("Generating earnings effect plot...")
    plot_earnings_effect(
        df=df,
        save_path=PLOTS_DIR / "earnings_effect.png",
    )

    print("Generating model predictions plot (dummy)...")
    dummy_true = df["rel_error"].values
    noise = np.random.default_rng(42).normal(0, 0.02, size=len(dummy_true))
    dummy_pred = dummy_true + noise
    plot_model_predictions(
        y_true=dummy_true,
        y_pred=dummy_pred,
        model_name="XGBoost (dummy)",
        save_path=PLOTS_DIR / "model_predictions_dummy.png",
    )

    print(f"\nAll plots saved to {PLOTS_DIR}")
