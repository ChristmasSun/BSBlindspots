import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import xgboost as xgb


def compute_shap_values(
    model: xgb.XGBRegressor | object,
    X: pd.DataFrame,
    model_type: str = "xgboost",
) -> shap.Explanation:
    if model_type in ("xgboost", "lightgbm"):
        explainer = shap.TreeExplainer(model)
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")
    shap_values = explainer(X)
    return shap_values


def plot_summary(
    shap_values: shap.Explanation,
    X: pd.DataFrame,
    save_path: str | None = None,
) -> None:
    plt.figure()
    shap.summary_plot(shap_values, X, show=False)
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def get_top_features(
    shap_values: shap.Explanation,
    X: pd.DataFrame,
    n: int = 10,
) -> list[tuple[str, float]]:
    mean_abs = np.abs(shap_values.values).mean(axis=0)
    feature_names = list(X.columns)
    pairs = []
    for i in range(len(feature_names)):
        pairs.append((feature_names[i], float(mean_abs[i])))
    pairs.sort(key=lambda x: x[1], reverse=True)
    return pairs[:n]


def plot_top_features(
    shap_values: shap.Explanation,
    X: pd.DataFrame,
    n_top: int = 5,
    save_dir: str | None = None,
) -> None:
    top = get_top_features(shap_values, X, n=n_top)
    for feat_name, _ in top:
        plt.figure()
        shap.dependence_plot(feat_name, shap_values.values, X, show=False)
        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)
            plt.savefig(
                os.path.join(save_dir, f"dependence_{feat_name}.png"),
                dpi=150,
                bbox_inches="tight",
            )
        plt.close()


def plot_interaction(
    shap_values: shap.Explanation,
    X: pd.DataFrame,
    feature1: str = "moneyness",
    feature2: str = "vix_regime",
    save_path: str | None = None,
) -> None:
    plt.figure()
    shap.dependence_plot(
        feature1,
        shap_values.values,
        X,
        interaction_index=feature2,
        show=False,
    )
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def analyze_by_option_type(
    model: xgb.XGBRegressor | object,
    X: pd.DataFrame,
    option_type_col: str = "option_type_binary",
    model_type: str = "xgboost",
    save_dir: str | None = None,
) -> dict:
    calls_mask = X[option_type_col] == 1
    puts_mask = X[option_type_col] == 0

    X_calls = X[calls_mask]
    X_puts = X[puts_mask]

    shap_calls = compute_shap_values(model, X_calls, model_type=model_type)
    shap_puts = compute_shap_values(model, X_puts, model_type=model_type)

    top_calls = get_top_features(shap_calls, X_calls)
    top_puts = get_top_features(shap_puts, X_puts)

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        plot_summary(
            shap_calls,
            X_calls,
            save_path=os.path.join(save_dir, "summary_calls.png"),
        )
        plot_summary(
            shap_puts,
            X_puts,
            save_path=os.path.join(save_dir, "summary_puts.png"),
        )

    return {
        "calls": top_calls,
        "puts": top_puts,
    }


def generate_full_report(
    model: xgb.XGBRegressor | object,
    X: pd.DataFrame,
    model_type: str = "xgboost",
) -> None:
    base_dir = Path("data/features/shap_plots")
    base_dir.mkdir(parents=True, exist_ok=True)

    print("Computing SHAP values...")
    shap_values = compute_shap_values(model, X, model_type=model_type)

    print("Generating summary plot...")
    plot_summary(
        shap_values,
        X,
        save_path=str(base_dir / "summary.png"),
    )

    print("Generating dependence plots for top 5 features...")
    plot_top_features(
        shap_values,
        X,
        n_top=5,
        save_dir=str(base_dir / "dependence"),
    )

    print("Generating interaction plot (moneyness x vix_regime)...")
    interaction_f1 = "moneyness"
    interaction_f2 = "vix_regime"
    cols = list(X.columns)
    if interaction_f1 in cols and interaction_f2 in cols:
        plot_interaction(
            shap_values,
            X,
            feature1=interaction_f1,
            feature2=interaction_f2,
            save_path=str(base_dir / "interaction_moneyness_vix_regime.png"),
        )
    else:
        print(f"  Skipping interaction plot: {interaction_f1} or {interaction_f2} not in features")

    print("Analyzing calls vs puts...")
    option_type_col = "option_type_binary"
    if option_type_col in cols:
        result = analyze_by_option_type(
            model,
            X,
            option_type_col=option_type_col,
            model_type=model_type,
            save_dir=str(base_dir / "by_option_type"),
        )
    else:
        print(f"  Skipping option type analysis: {option_type_col} not in features")
        result = {}

    top = get_top_features(shap_values, X, n=10)

    print("\n" + "=" * 60)
    print("SHAP ANALYSIS SUMMARY")
    print("=" * 60)

    print("\nTop 10 features by mean |SHAP|:")
    for rank, (name, val) in enumerate(top, 1):
        print(f"  {rank:2d}. {name:<30s} {val:.6f}")

    earnings_features = ["days_to_earnings", "days_since_earnings", "in_earnings_window"]
    found_earnings = []
    for feat_name, feat_val in top:
        if feat_name in earnings_features:
            found_earnings.append((feat_name, feat_val))
    if found_earnings:
        print("\nEarnings features in top 10:")
        for name, val in found_earnings:
            print(f"  - {name}: {val:.6f}")
    else:
        print("\nNo earnings features found in top 10.")

    if result:
        print("\nTop 5 features for CALLS:")
        for rank, (name, val) in enumerate(result["calls"][:5], 1):
            print(f"  {rank:2d}. {name:<30s} {val:.6f}")
        print("\nTop 5 features for PUTS:")
        for rank, (name, val) in enumerate(result["puts"][:5], 1):
            print(f"  {rank:2d}. {name:<30s} {val:.6f}")

    print(f"\nPlots saved to {base_dir.resolve()}")
    print("=" * 60)


if __name__ == "__main__":
    import joblib

    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    features_path = PROJECT_ROOT / "data" / "features" / "SPY_features.csv"
    models_dir = PROJECT_ROOT / "data" / "models"
    model_path = models_dir / "xgboost.json"

    if not features_path.exists():
        print(f"Features file not found: {features_path}")
        print("Run src/features/build_features.py first.")
        raise SystemExit(1)

    if not model_path.exists():
        print(f"Model file not found: {model_path}")
        print("Run src/models/train.py first.")
        raise SystemExit(1)

    print(f"Loading features from {features_path}...")
    df = pd.read_csv(features_path)

    target_col = "rel_error"
    drop_cols = []
    for col in [target_col, "abs_error", "signed_error", "sq_error", "mid_price", "bs_price",
                "underlying_price", "bid", "ask"]:
        if col in df.columns:
            drop_cols.append(col)
    X = df.drop(columns=drop_cols)

    date_cols = []
    for col in X.columns:
        if pd.api.types.is_datetime64_any_dtype(X[col]):
            date_cols.append(col)
    if date_cols:
        X = X.drop(columns=date_cols)

    str_cols = []
    for col in X.columns:
        if X[col].dtype == object:
            str_cols.append(col)
    if str_cols:
        X = X.drop(columns=str_cols)

    print(f"Loading model from {model_path}...")
    model = xgb.XGBRegressor()
    model.load_model(str(model_path))

    generate_full_report(model, X)
