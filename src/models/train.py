from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def get_feature_columns() -> list[str]:
    columns = [
        "moneyness",
        "log_moneyness",
        "dte",
        "time_to_maturity",
        "option_type_binary",
        "hist_vol_20",
        "hist_vol_60",
        "vol_ratio",
        "vix",
        "vol_of_vol",
        "days_to_earnings",
        "days_since_earnings",
        "in_earnings_window",
        "bid_ask_spread",
        "bid_ask_rel",
        "log_volume",
        "log_open_interest",
        "vix_regime",
    ]
    return columns


def load_features(ticker: str) -> pd.DataFrame:
    path = PROJECT_ROOT / "data" / "features" / f"{ticker}_features.csv"
    df = pd.read_csv(path, parse_dates=["date"])
    return df


def time_split(
    df: pd.DataFrame, test_fraction: float = 0.2
) -> tuple[pd.DataFrame, pd.DataFrame]:
    df_sorted = df.sort_values("date").reset_index(drop=True)
    cutoff_idx = int(len(df_sorted) * (1 - test_fraction))
    train_df = df_sorted.iloc[:cutoff_idx].copy()
    test_df = df_sorted.iloc[cutoff_idx:].copy()
    return train_df, test_df


def train_xgboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    params: dict | None = None,
) -> xgb.XGBRegressor:
    default_params = {
        "n_estimators": 500,
        "max_depth": 6,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": 42,
        "n_jobs": -1,
    }
    if params is not None:
        default_params.update(params)
    model = xgb.XGBRegressor(**default_params)
    model.fit(X_train, y_train)
    return model


def train_lightgbm(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    params: dict | None = None,
) -> lgb.LGBMRegressor:
    default_params = {
        "n_estimators": 500,
        "max_depth": 6,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": 42,
        "n_jobs": -1,
        "verbosity": -1,
    }
    if params is not None:
        default_params.update(params)
    model = lgb.LGBMRegressor(**default_params)
    model.fit(X_train, y_train)
    return model


def cross_validate_timeseries(
    df: pd.DataFrame,
    model_type: str = "xgboost",
    n_splits: int = 5,
) -> dict[str, float]:
    feature_cols = get_feature_columns()
    df_sorted = df.sort_values("date").reset_index(drop=True)
    X = df_sorted[feature_cols]
    y = df_sorted["rel_error"]

    tscv = TimeSeriesSplit(n_splits=n_splits)

    mae_scores = []
    rmse_scores = []
    r2_scores = []

    for train_idx, test_idx in tscv.split(X):
        X_train = X.iloc[train_idx]
        X_test = X.iloc[test_idx]
        y_train = y.iloc[train_idx]
        y_test = y.iloc[test_idx]

        if model_type == "xgboost":
            model = train_xgboost(X_train, y_train)
        else:
            model = train_lightgbm(X_train, y_train)

        preds = model.predict(X_test)

        mae_scores.append(mean_absolute_error(y_test, preds))
        rmse_scores.append(np.sqrt(mean_squared_error(y_test, preds)))
        r2_scores.append(r2_score(y_test, preds))

    results = {
        "mae_mean": np.mean(mae_scores),
        "mae_std": np.std(mae_scores),
        "rmse_mean": np.mean(rmse_scores),
        "rmse_std": np.std(rmse_scores),
        "r2_mean": np.mean(r2_scores),
        "r2_std": np.std(r2_scores),
    }
    return results


def save_model(model: xgb.XGBRegressor | lgb.LGBMRegressor, name: str) -> None:
    models_dir = PROJECT_ROOT / "data" / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    if isinstance(model, xgb.XGBRegressor):
        path = models_dir / f"{name}.json"
        model.save_model(str(path))
    else:
        path = models_dir / f"{name}.txt"
        model.booster_.save_model(str(path))

    print(f"Model saved to {path}")


def evaluate_model(
    model: xgb.XGBRegressor | lgb.LGBMRegressor,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict[str, float]:
    preds = model.predict(X_test)
    results = {
        "mae": mean_absolute_error(y_test, preds),
        "rmse": np.sqrt(mean_squared_error(y_test, preds)),
        "r2": r2_score(y_test, preds),
    }
    return results


if __name__ == "__main__":
    print("Loading SPY features...")
    df = load_features("SPY")
    print(f"Loaded {len(df)} rows")

    feature_cols = get_feature_columns()
    train_df, test_df = time_split(df)
    print(f"Train: {len(train_df)} rows, Test: {len(test_df)} rows")

    X_train = train_df[feature_cols]
    y_train = train_df["rel_error"]
    X_test = test_df[feature_cols]
    y_test = test_df["rel_error"]

    print("\nTraining XGBoost...")
    xgb_model = train_xgboost(X_train, y_train)
    xgb_results = evaluate_model(xgb_model, X_test, y_test)
    print(f"  MAE:  {xgb_results['mae']:.6f}")
    print(f"  RMSE: {xgb_results['rmse']:.6f}")
    print(f"  R²:   {xgb_results['r2']:.6f}")

    print("\nTraining LightGBM...")
    lgb_model = train_lightgbm(X_train, y_train)
    lgb_results = evaluate_model(lgb_model, X_test, y_test)
    print(f"  MAE:  {lgb_results['mae']:.6f}")
    print(f"  RMSE: {lgb_results['rmse']:.6f}")
    print(f"  R²:   {lgb_results['r2']:.6f}")

    print("\nCross-validating XGBoost (TimeSeriesSplit)...")
    xgb_cv = cross_validate_timeseries(df, model_type="xgboost", n_splits=5)
    print(f"  MAE:  {xgb_cv['mae_mean']:.6f} ± {xgb_cv['mae_std']:.6f}")
    print(f"  RMSE: {xgb_cv['rmse_mean']:.6f} ± {xgb_cv['rmse_std']:.6f}")
    print(f"  R²:   {xgb_cv['r2_mean']:.6f} ± {xgb_cv['r2_std']:.6f}")

    print("\nCross-validating LightGBM (TimeSeriesSplit)...")
    lgb_cv = cross_validate_timeseries(df, model_type="lightgbm", n_splits=5)
    print(f"  MAE:  {lgb_cv['mae_mean']:.6f} ± {lgb_cv['mae_std']:.6f}")
    print(f"  RMSE: {lgb_cv['rmse_mean']:.6f} ± {lgb_cv['rmse_std']:.6f}")
    print(f"  R²:   {lgb_cv['r2_mean']:.6f} ± {lgb_cv['r2_std']:.6f}")

    save_model(xgb_model, "xgb_spy")
    save_model(lgb_model, "lgb_spy")

    print("\nDone.")
