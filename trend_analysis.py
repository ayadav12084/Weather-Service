import logging
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

from db import get_connection

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Raw data loader
# ---------------------------------------------------------------------------

def load_readings(
    region_id: int,
    days: int = 30,
    end_dt: datetime | None = None,
) -> pd.DataFrame:
    """
    Load hourly weather readings for a region.

    Returns a DataFrame indexed by `recorded_at` (UTC, timezone-aware),
    sorted ascending. Columns match weather_readings exactly.
    """
    if end_dt is None:
        end_dt = datetime.now(tz=timezone.utc)
    start_dt = end_dt - timedelta(days=days)

    sql = """
        SELECT
            wr.recorded_at,
            wr.temp_celsius,
            wr.feels_like,
            wr.humidity_pct,
            wr.pressure_hpa,
            wr.wind_speed_ms,
            wr.wind_dir_deg,
            wr.precipitation_mm,
            wr.cloud_cover_pct,
            wc.category  AS condition_category
        FROM weather_readings wr
        LEFT JOIN weather_conditions wc ON wc.id = wr.condition_id
        WHERE wr.region_id  = %s
          AND wr.recorded_at >= %s
          AND wr.recorded_at <  %s
        ORDER BY wr.recorded_at ASC
    """
    with get_connection() as conn:
        df = pd.read_sql(sql, conn, params=(region_id, start_dt, end_dt),
                         parse_dates=["recorded_at"])

    if df.empty:
        logger.warning("No readings for region_id=%s in last %d days", region_id, days)
        return df

    df["recorded_at"] = pd.to_datetime(df["recorded_at"], utc=True)
    df = df.set_index("recorded_at").sort_index()

    # Forward-fill short gaps (up to 2 hours) then drop remaining NaNs
    numeric_cols = df.select_dtypes(include="number").columns
    df[numeric_cols] = (
        df[numeric_cols]
        .resample("h")           # ensure regular hourly grid
        .mean()
        .ffill(limit=2)
    )
    logger.info("Loaded %d hourly rows for region_id=%s", len(df), region_id)
    return df


# ---------------------------------------------------------------------------
# Daily aggregations
# ---------------------------------------------------------------------------

def daily_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate an hourly DataFrame to daily statistics.

    Returned columns:
        temp_min, temp_max, temp_mean, temp_range,
        humidity_mean, pressure_mean,
        wind_max, precipitation_total,
        dominant_condition
    """
    if df.empty:
        return df

    agg = df.resample("D").agg(
        temp_min        =("temp_celsius",     "min"),
        temp_max        =("temp_celsius",     "max"),
        temp_mean       =("temp_celsius",     "mean"),
        humidity_mean   =("humidity_pct",     "mean"),
        pressure_mean   =("pressure_hpa",     "mean"),
        wind_max        =("wind_speed_ms",    "max"),
        precipitation_total=("precipitation_mm", "sum"),
        dominant_condition=("condition_category",
                            lambda s: s.mode().iloc[0] if not s.mode().empty else None),
    )
    agg["temp_range"] = agg["temp_max"] - agg["temp_min"]
    return agg.round(2)


# ---------------------------------------------------------------------------
# Rolling / moving average
# ---------------------------------------------------------------------------

def rolling_temperature(df: pd.DataFrame, window_hours: int = 24) -> pd.DataFrame:
    """
    Return a DataFrame with rolling mean and std of temperature.
    Useful for smoothed trend lines on charts.
    """
    if df.empty:
        return df

    result = pd.DataFrame(index=df.index)
    result["temp_celsius"]    = df["temp_celsius"]
    result["rolling_mean"]    = df["temp_celsius"].rolling(window_hours, min_periods=1).mean()
    result["rolling_std"]     = df["temp_celsius"].rolling(window_hours, min_periods=1).std()
    result["rolling_upper"]   = result["rolling_mean"] + result["rolling_std"]
    result["rolling_lower"]   = result["rolling_mean"] - result["rolling_std"]
    return result.round(3)


# ---------------------------------------------------------------------------
# Anomaly detection (Z-score)
# ---------------------------------------------------------------------------

def detect_anomalies(df: pd.DataFrame, column: str = "temp_celsius",
                     z_threshold: float = 2.5) -> pd.DataFrame:
    """
    Flag readings where the column value deviates more than z_threshold
    standard deviations from the rolling 7-day mean.

    Returns the original DataFrame with two extra columns:
        rolling_mean_7d, z_score, is_anomaly
    """
    if df.empty or column not in df.columns:
        return df

    window = 7 * 24   # 7 days in hours
    rolling_mean = df[column].rolling(window, min_periods=24).mean()
    rolling_std  = df[column].rolling(window, min_periods=24).std()

    out = df.copy()
    out["rolling_mean_7d"] = rolling_mean
    out["z_score"]         = (df[column] - rolling_mean) / rolling_std.replace(0, np.nan)
    out["is_anomaly"]      = out["z_score"].abs() > z_threshold
    return out


# ---------------------------------------------------------------------------
# Seasonal decomposition helper
# ---------------------------------------------------------------------------

def temperature_decomposition(df: pd.DataFrame) -> dict:
    """
    Decompose hourly temperature into trend + seasonal + residual
    using a simple additive STL-style approach with Pandas.

    Returns a dict of DataFrames: {trend, seasonal, residual}.
    Requires at least 48 hours of data.
    """
    if len(df) < 48:
        raise ValueError("Need at least 48 hours of data for decomposition")

    series = df["temp_celsius"].dropna()

    # 24-hour centred moving average → trend
    trend    = series.rolling(24, center=True, min_periods=12).mean()
    detrended = series - trend

    # Average by hour-of-day → seasonal component
    hour_means = detrended.groupby(detrended.index.hour).mean()
    seasonal   = detrended.index.map(lambda ts: hour_means.get(ts.hour, 0))
    seasonal   = pd.Series(seasonal.values, index=detrended.index, name="seasonal")

    residual   = detrended - seasonal

    return {
        "trend":    trend.rename("trend"),
        "seasonal": seasonal,
        "residual": residual.rename("residual"),
    }


# ---------------------------------------------------------------------------
# Feature engineering  (used by predictor.py)
# ---------------------------------------------------------------------------

def build_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineer lag and time features from an hourly DataFrame.

    Output columns (all numeric, no NaNs after dropna):
        hour_sin, hour_cos           — cyclic hour encoding
        dow_sin, dow_cos             — cyclic day-of-week encoding
        month_sin, month_cos         — cyclic month encoding
        temp_lag_1h  … temp_lag_24h  — lagged temperature
        temp_roll_mean_6h            — 6-hour rolling mean temp
        temp_roll_mean_24h           — 24-hour rolling mean temp
        humidity_lag_1h              — lagged humidity
        pressure_lag_1h              — lagged pressure
        precip_roll_sum_6h           — 6-hour rolling precipitation sum
        temp_celsius                 — TARGET (next hour's temperature)
    """
    if df.empty:
        return df

    feat = pd.DataFrame(index=df.index)

    # --- Cyclic time encodings ---
    hour  = df.index.hour
    dow   = df.index.dayofweek
    month = df.index.month

    feat["hour_sin"]  = np.sin(2 * np.pi * hour  / 24)
    feat["hour_cos"]  = np.cos(2 * np.pi * hour  / 24)
    feat["dow_sin"]   = np.sin(2 * np.pi * dow   / 7)
    feat["dow_cos"]   = np.cos(2 * np.pi * dow   / 7)
    feat["month_sin"] = np.sin(2 * np.pi * month / 12)
    feat["month_cos"] = np.cos(2 * np.pi * month / 12)

    # --- Temperature lags ---
    for lag in [1, 3, 6, 12, 24]:
        feat[f"temp_lag_{lag}h"] = df["temp_celsius"].shift(lag)

    # --- Rolling means ---
    feat["temp_roll_mean_6h"]  = df["temp_celsius"].rolling(6,  min_periods=1).mean()
    feat["temp_roll_mean_24h"] = df["temp_celsius"].rolling(24, min_periods=1).mean()

    # --- Other feature lags ---
    feat["humidity_lag_1h"]    = df["humidity_pct"].shift(1)
    feat["pressure_lag_1h"]    = df["pressure_hpa"].shift(1)
    feat["precip_roll_sum_6h"] = df["precipitation_mm"].rolling(6, min_periods=1).sum()

    # --- Target: temperature at this timestep (predict from prior lags) ---
    feat["temp_celsius"] = df["temp_celsius"]

    return feat.dropna()


# ---------------------------------------------------------------------------
# Convenience: chart-ready JSON payload
# ---------------------------------------------------------------------------

def chart_payload(region_id: int, days: int = 7) -> dict:
    """
    Return a dict ready to be JSON-serialised for the frontend charts.

    {
      "labels":       ["2025-03-10T00:00Z", ...],   # ISO strings
      "temp":         [12.3, ...],
      "temp_upper":   [13.1, ...],
      "temp_lower":   [11.5, ...],
      "humidity":     [72.0, ...],
      "precipitation":[0.0, ...],
      "daily": {
        "labels":        ["2025-03-10", ...],
        "temp_min":      [...],
        "temp_max":      [...],
        "precipitation": [...],
      }
    }
    """
    df = load_readings(region_id, days=days)
    if df.empty:
        return {}

    rolled = rolling_temperature(df)
    daily  = daily_summary(df)

    def _ts(index):
        return [t.isoformat() for t in index]

    return {
        "labels":        _ts(rolled.index),
        "temp":          rolled["temp_celsius"].round(2).tolist(),
        "temp_upper":    rolled["rolling_upper"].round(2).tolist(),
        "temp_lower":    rolled["rolling_lower"].round(2).tolist(),
        "humidity":      df["humidity_pct"].round(1).tolist(),
        "precipitation": df["precipitation_mm"].round(2).tolist(),
        "daily": {
            "labels":        [d.date().isoformat() for d in daily.index],
            "temp_min":      daily["temp_min"].tolist(),
            "temp_max":      daily["temp_max"].tolist(),
            "precipitation": daily["precipitation_total"].tolist(),
        },
    }
