import logging
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit

from db import get_connection, bulk_insert_readings
from trend_analysis import load_readings, build_feature_matrix

logger = logging.getLogger(__name__)

# How many days of history to train on
TRAINING_DAYS = 60

# Minimum rows needed before we attempt training
MIN_TRAINING_ROWS = 48


# ---------------------------------------------------------------------------
# DB write helpers
# ---------------------------------------------------------------------------

def _save_model_run(region_id: int, model_name: str, n_rows: int,
                    mae: float, rmse: float, horizon_days: int) -> int:
    """Insert a model_runs record and return its id."""
    sql = """
        INSERT INTO model_runs
            (region_id, model_name, training_rows, mae, rmse, horizon_days)
        VALUES (%s, %s, %s, %s, %s, %s)
    """
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (region_id, model_name,
                              n_rows, round(mae, 4), round(rmse, 4), horizon_days))
            run_id = cur.lastrowid
        conn.commit()
    return run_id


def _save_forecasts(region_id: int, model_run_id: int, rows: list[dict]) -> None:
    """Bulk-insert forecast rows."""
    sql = """
        INSERT INTO forecasts
            (region_id, model_run_id, forecast_for, generated_at,
             temp_min, temp_max, temp_mean,
             precip_prob, wind_speed_ms,
             confidence_lo, confidence_hi)
        VALUES
            (%(region_id)s, %(model_run_id)s, %(forecast_for)s, %(generated_at)s,
             %(temp_min)s, %(temp_max)s, %(temp_mean)s,
             %(precip_prob)s, %(wind_speed_ms)s,
             %(confidence_lo)s, %(confidence_hi)s)
        ON DUPLICATE KEY UPDATE
            temp_mean      = VALUES(temp_mean),
            confidence_lo  = VALUES(confidence_lo),
            confidence_hi  = VALUES(confidence_hi)
    """
    now = datetime.now(tz=timezone.utc)
    tagged = [{**r, "region_id": region_id,
               "model_run_id": model_run_id,
               "generated_at": now} for r in rows]

    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.executemany(sql, tagged)
        conn.commit()

    logger.info("Saved %d forecast rows (model_run_id=%d)", len(rows), model_run_id)


# ---------------------------------------------------------------------------
# Model 1: Prophet
# ---------------------------------------------------------------------------

def _forecast_prophet(df: pd.DataFrame, horizon_days: int) -> tuple[list[dict], float, float]:
    """
    Fit Facebook Prophet on daily temperature and produce a horizon_days forecast.
    Returns (forecast_rows, mae, rmse).
    """
    try:
        from prophet import Prophet
    except ImportError as exc:
        raise ImportError(
            "prophet is not installed. Run: pip install prophet"
        ) from exc

    # Prophet expects a DataFrame with columns ds (datetime) and y (value)
    daily = df["temp_celsius"].resample("D").mean().dropna()
    prophet_df = pd.DataFrame({
        "ds": daily.index.tz_localize(None),   # Prophet requires tz-naive
        "y":  daily.values,
    })

    if len(prophet_df) < 14:
        raise ValueError("Prophet needs at least 14 days of daily data")

    # --- Train / validation split (last 7 days held out) ---
    split = len(prophet_df) - 7
    train_df = prophet_df.iloc[:split]
    val_df   = prophet_df.iloc[split:]

    model = Prophet(
        daily_seasonality=True,
        weekly_seasonality=True,
        yearly_seasonality=False,   # enable once you have >1 year of data
        interval_width=0.80,        # 80% confidence interval
        changepoint_prior_scale=0.05,
    )
    model.fit(train_df)

    # Validate on held-out week
    val_forecast = model.predict(val_df[["ds"]])
    mae  = mean_absolute_error(val_df["y"], val_forecast["yhat"])
    rmse = mean_squared_error(val_df["y"],  val_forecast["yhat"], squared=False)
    logger.info("Prophet validation — MAE=%.3f°C  RMSE=%.3f°C", mae, rmse)

    # Re-fit on all data for production forecast
    model.fit(prophet_df)
    future = model.make_future_dataframe(periods=horizon_days, freq="D")
    forecast = model.predict(future).tail(horizon_days)

    rows = []
    for _, row in forecast.iterrows():
        rows.append({
            "forecast_for":  row["ds"].replace(tzinfo=timezone.utc),
            "temp_min":      round(row["yhat_lower"], 2),
            "temp_max":      round(row["yhat_upper"], 2),
            "temp_mean":     round(row["yhat"], 2),
            "precip_prob":   None,
            "wind_speed_ms": None,
            "confidence_lo": round(row["yhat_lower"], 2),
            "confidence_hi": round(row["yhat_upper"], 2),
        })

    return rows, mae, rmse


# ---------------------------------------------------------------------------
# Model 2: Random Forest  (short-horizon, hourly)
# ---------------------------------------------------------------------------

def _forecast_random_forest(df: pd.DataFrame, horizon_days: int) -> tuple[list[dict], float, float]:
    """
    Train a Random Forest on lag/time features and forecast the next
    horizon_days × 24 hourly temperature values.

    Returns (forecast_rows aggregated to daily, mae, rmse).
    """
    feat_df = build_feature_matrix(df)
    if len(feat_df) < MIN_TRAINING_ROWS:
        raise ValueError(f"Need at least {MIN_TRAINING_ROWS} rows; got {len(feat_df)}")

    feature_cols = [c for c in feat_df.columns if c != "temp_celsius"]
    X = feat_df[feature_cols].values
    y = feat_df["temp_celsius"].values

    # --- Time-series cross-validation (3 folds) ---
    tscv = TimeSeriesSplit(n_splits=3)
    val_maes, val_rmses = [], []

    for train_idx, val_idx in tscv.split(X):
        model = RandomForestRegressor(
            n_estimators=200,
            max_depth=12,
            min_samples_leaf=4,
            random_state=42,
            n_jobs=-1,
        )
        model.fit(X[train_idx], y[train_idx])
        preds = model.predict(X[val_idx])
        val_maes.append(mean_absolute_error(y[val_idx], preds))
        val_rmses.append(mean_squared_error(y[val_idx], preds, squared=False))

    mae  = float(np.mean(val_maes))
    rmse = float(np.mean(val_rmses))
    logger.info("RandomForest CV — MAE=%.3f°C  RMSE=%.3f°C", mae, rmse)

    # --- Final fit on all data ---
    model = RandomForestRegressor(
        n_estimators=200, max_depth=12,
        min_samples_leaf=4, random_state=42, n_jobs=-1,
    )
    model.fit(X, y)

    # --- Recursive multi-step forecast ---
    # Build the initial feature row from the last known row in feat_df
    last_known = feat_df.iloc[-1].copy()
    last_temp  = df["temp_celsius"].iloc[-1]
    last_ts    = df.index[-1]

    hourly_preds = []
    temp_history = list(df["temp_celsius"].dropna().values[-24:])   # sliding window

    for step in range(horizon_days * 24):
        ts = last_ts + timedelta(hours=step + 1)
        hour  = ts.hour
        dow   = ts.dayofweek
        month = ts.month

        row = {
            "hour_sin":           np.sin(2 * np.pi * hour  / 24),
            "hour_cos":           np.cos(2 * np.pi * hour  / 24),
            "dow_sin":            np.sin(2 * np.pi * dow   / 7),
            "dow_cos":            np.cos(2 * np.pi * dow   / 7),
            "month_sin":          np.sin(2 * np.pi * month / 12),
            "month_cos":          np.cos(2 * np.pi * month / 12),
            "temp_lag_1h":        temp_history[-1]  if len(temp_history) >= 1  else last_temp,
            "temp_lag_3h":        temp_history[-3]  if len(temp_history) >= 3  else last_temp,
            "temp_lag_6h":        temp_history[-6]  if len(temp_history) >= 6  else last_temp,
            "temp_lag_12h":       temp_history[-12] if len(temp_history) >= 12 else last_temp,
            "temp_lag_24h":       temp_history[-24] if len(temp_history) >= 24 else last_temp,
            "temp_roll_mean_6h":  float(np.mean(temp_history[-6:])),
            "temp_roll_mean_24h": float(np.mean(temp_history[-24:])),
            "humidity_lag_1h":    df["humidity_pct"].iloc[-1],
            "pressure_lag_1h":    df["pressure_hpa"].iloc[-1],
            "precip_roll_sum_6h": float(df["precipitation_mm"].iloc[-6:].sum()),
        }

        X_pred   = np.array([[row[c] for c in feature_cols]])
        pred_val = float(model.predict(X_pred)[0])

        # Estimate uncertainty from training residuals (simple std of errors)
        residual_std = rmse
        hourly_preds.append({
            "ts":    ts,
            "temp":  pred_val,
            "upper": pred_val + 1.28 * residual_std,   # ~80% CI
            "lower": pred_val - 1.28 * residual_std,
        })
        temp_history.append(pred_val)

    # Aggregate hourly predictions to daily
    pred_df = pd.DataFrame(hourly_preds).set_index("ts")
    daily   = pred_df.resample("D").agg(
        temp_min  =("lower", "min"),
        temp_max  =("upper", "max"),
        temp_mean =("temp",  "mean"),
    ).round(2)

    rows = []
    for ts, row in daily.iterrows():
        rows.append({
            "forecast_for":  ts.replace(tzinfo=timezone.utc),
            "temp_min":      row["temp_min"],
            "temp_max":      row["temp_max"],
            "temp_mean":     row["temp_mean"],
            "precip_prob":   None,
            "wind_speed_ms": None,
            "confidence_lo": row["temp_min"],
            "confidence_hi": row["temp_max"],
        })

    return rows, mae, rmse


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

_MODEL_REGISTRY = {
    "prophet":       _forecast_prophet,
    "random_forest": _forecast_random_forest,
}


def run_forecast(
    region_id: int,
    horizon_days: int = 5,
    model: str = "prophet",
) -> dict:
    """
    Train a model and write forecast + model_run rows to the DB.

    Returns a summary dict:
    {
        "model_run_id": int,
        "model":        str,
        "horizon_days": int,
        "mae":          float,
        "rmse":         float,
        "rows_written": int,
    }
    """
    if model not in _MODEL_REGISTRY:
        raise ValueError(f"Unknown model '{model}'. Choose from: {list(_MODEL_REGISTRY)}")

    logger.info("Starting %s forecast for region_id=%d, horizon=%dd",
                model, region_id, horizon_days)

    df = load_readings(region_id, days=TRAINING_DAYS)
    if df.empty or len(df) < MIN_TRAINING_ROWS:
        raise RuntimeError(
            f"Insufficient data for region_id={region_id}: "
            f"found {len(df)} rows, need {MIN_TRAINING_ROWS}"
        )

    forecast_fn = _MODEL_REGISTRY[model]
    rows, mae, rmse = forecast_fn(df, horizon_days)

    run_id = _save_model_run(region_id, model, len(df), mae, rmse, horizon_days)
    _save_forecasts(region_id, run_id, rows)

    logger.info("Forecast complete — model_run_id=%d, MAE=%.3f°C", run_id, mae)
    return {
        "model_run_id": run_id,
        "model":        model,
        "horizon_days": horizon_days,
        "mae":          round(mae, 3),
        "rmse":         round(rmse, 3),
        "rows_written": len(rows),
    }


def load_latest_forecast(region_id: int, horizon_days: int = 5) -> pd.DataFrame:
    """
    Load the most recent forecast for a region from the DB.

    Returns a DataFrame with columns:
        forecast_for, temp_min, temp_max, temp_mean,
        confidence_lo, confidence_hi, model_name
    """
    sql = """
        SELECT
            f.forecast_for,
            f.temp_min, f.temp_max, f.temp_mean,
            f.confidence_lo, f.confidence_hi,
            mr.model_name,
            mr.mae
        FROM forecasts f
        JOIN model_runs mr ON mr.id = f.model_run_id
        WHERE f.region_id = %s
          AND f.forecast_for >= NOW()
          AND mr.id = (
              SELECT id FROM model_runs
              WHERE region_id = %s
              ORDER BY run_at DESC LIMIT 1
          )
        ORDER BY f.forecast_for ASC
        LIMIT %s
    """
    with get_connection() as conn:
        df = pd.read_sql(sql, conn,
                         params=(region_id, region_id, horizon_days),
                         parse_dates=["forecast_for"])

    if not df.empty:
        df["forecast_for"] = pd.to_datetime(df["forecast_for"], utc=True)

    return df


def forecast_payload(region_id: int, horizon_days: int = 5) -> dict:
    """
    Return a dict ready to be JSON-serialised for the frontend forecast panel.

    {
      "labels":        ["2025-03-17", ...],
      "temp_mean":     [12.3, ...],
      "temp_min":      [9.1, ...],
      "temp_max":      [15.6, ...],
      "confidence_lo": [8.0, ...],
      "confidence_hi": [16.9, ...],
      "model_name":    "prophet",
      "mae":           0.84,
    }
    """
    df = load_latest_forecast(region_id, horizon_days)
    if df.empty:
        return {}

    def _fmt(col):
        return df[col].round(2).tolist() if col in df.columns else []

    return {
        "labels":        [d.date().isoformat() for d in df["forecast_for"]],
        "temp_mean":     _fmt("temp_mean"),
        "temp_min":      _fmt("temp_min"),
        "temp_max":      _fmt("temp_max"),
        "confidence_lo": _fmt("confidence_lo"),
        "confidence_hi": _fmt("confidence_hi"),
        "model_name":    df["model_name"].iloc[0] if not df.empty else None,
        "mae":           round(float(df["mae"].iloc[0]), 3) if not df.empty else None,
    }
