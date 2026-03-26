import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score


PROJECT_ROOT = Path(__file__).resolve().parents[2]
FEATURES_DIR = PROJECT_ROOT / "data" / "features"


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    errors = []
    sq_errors = []
    for i in range(len(y_true)):
        diff = y_true[i] - y_pred[i]
        errors.append(abs(diff))
        sq_errors.append(diff ** 2)

    mae = sum(errors) / len(errors)

    mse = sum(sq_errors) / len(sq_errors)
    rmse = math.sqrt(mse)

    r2 = r2_score(y_true, y_pred)

    return {"mae": mae, "rmse": rmse, "r2": r2}


def compute_baseline_metrics(y_true: np.ndarray) -> dict:
    baseline_pred = np.zeros(len(y_true))
    return compute_metrics(y_true, baseline_pred)


def evaluate_model(model: object, X_test: pd.DataFrame, y_test: np.ndarray, model_name: str) -> dict:
    y_pred = model.predict(X_test)
    metrics = compute_metrics(y_test, y_pred)

    print(f"\n{'=' * 50}")
    print(f"  Model: {model_name}")
    print(f"{'=' * 50}")
    print(f"  MAE:   {metrics['mae']:.6f}")
    print(f"  RMSE:  {metrics['rmse']:.6f}")
    print(f"  R²:    {metrics['r2']:.6f}")
    print(f"{'=' * 50}\n")

    return metrics


def compare_models(results: dict[str, dict]) -> pd.DataFrame:
    rows = []
    for name, metrics in results.items():
        row = {"model": name}
        row["mae"] = metrics["mae"]
        row["rmse"] = metrics["rmse"]
        row["r2"] = metrics["r2"]
        rows.append(row)

    df = pd.DataFrame(rows)
    df = df.set_index("model")
    df = df.sort_values("mae", ascending=True)

    print("\n--- Model Comparison ---")
    print(df.to_string())
    print()

    return df


def evaluate_by_regime(
    model: object,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    regime_column: str,
) -> dict:
    results = {}
    unique_regimes = X_test[regime_column].unique()

    for regime in sorted(unique_regimes):
        mask = X_test[regime_column] == regime
        X_subset = X_test[mask]
        y_subset = y_test[mask]

        if len(y_subset) == 0:
            continue

        y_pred = model.predict(X_subset)
        metrics = compute_metrics(y_subset, y_pred)
        metrics["n_samples"] = len(y_subset)
        results[regime] = metrics

    print(f"\n--- Evaluation by {regime_column} ---")
    for regime, metrics in results.items():
        print(f"  {regime_column}={regime}  (n={metrics['n_samples']})")
        print(f"    MAE={metrics['mae']:.6f}  RMSE={metrics['rmse']:.6f}  R²={metrics['r2']:.6f}")
    print()

    return results


def evaluate_by_moneyness(
    model: object,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    bins: int = 5,
) -> dict:
    moneyness = X_test["moneyness"].copy()
    bin_labels = pd.qcut(moneyness, q=bins, duplicates="drop")

    results = {}
    unique_bins = bin_labels.unique()

    sorted_bins = sorted(unique_bins, key=lambda x: x.left)

    for b in sorted_bins:
        mask = bin_labels == b
        X_subset = X_test[mask]
        y_subset = y_test[mask]

        if len(y_subset) == 0:
            continue

        y_pred = model.predict(X_subset)
        metrics = compute_metrics(y_subset, y_pred)
        metrics["n_samples"] = len(y_subset)
        bin_label = f"{b.left:.3f}-{b.right:.3f}"
        results[bin_label] = metrics

    print("\n--- Evaluation by Moneyness Bin ---")
    for bin_label, metrics in results.items():
        print(f"  moneyness={bin_label}  (n={metrics['n_samples']})")
        print(f"    MAE={metrics['mae']:.6f}  RMSE={metrics['rmse']:.6f}  R²={metrics['r2']:.6f}")
    print()

    return results


if __name__ == "__main__":
    import joblib

    features_path = FEATURES_DIR / "SPY_features.csv"
    if not features_path.exists():
        print(f"Features file not found at {features_path}")
        print("Run src/features/build_features.py first.")
        raise SystemExit(1)

    df = pd.read_csv(features_path)
    df = df.sort_values("date").reset_index(drop=True)

    split_idx = int(len(df) * 0.8)
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]

    target_col = "rel_error"

    exclude_cols = [
        "date", "ticker", "expiration", "strike", "option_type",
        "abs_error", "signed_error", "rel_error", "sq_error",
        "mid_price", "bs_price", "bid", "ask", "underlying_price",
    ]

    feature_cols = []
    for col in df.columns:
        if col not in exclude_cols:
            feature_cols.append(col)

    X_train = train_df[feature_cols]
    y_train = train_df[target_col].values
    X_test = test_df[feature_cols]
    y_test = test_df[target_col].values

    models_dir = PROJECT_ROOT / "models"
    all_results = {}

    baseline_metrics = compute_baseline_metrics(y_test)
    all_results["baseline"] = baseline_metrics
    print("\n--- Baseline (BS alone, prediction = 0) ---")
    print(f"  MAE:  {baseline_metrics['mae']:.6f}")
    print(f"  RMSE: {baseline_metrics['rmse']:.6f}")
    print(f"  R²:   {baseline_metrics['r2']:.6f}")

    model_files = {
        "xgboost": models_dir / "xgboost_model.joblib",
        "lightgbm": models_dir / "lightgbm_model.joblib",
    }

    for name, model_path in model_files.items():
        if not model_path.exists():
            print(f"\nModel file not found: {model_path}")
            print(f"Skipping {name}. Run src/models/train.py first.")
            continue

        model = joblib.load(model_path)
        metrics = evaluate_model(model, X_test, y_test, name)
        all_results[name] = metrics

        if "vix_regime" in X_test.columns:
            evaluate_by_regime(model, X_test, y_test, "vix_regime")

        if "moneyness" in X_test.columns:
            evaluate_by_moneyness(model, X_test, y_test)

    if len(all_results) > 1:
        compare_models(all_results)
