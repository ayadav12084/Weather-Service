import logging
from contextlib import contextmanager

import mysql.connector
from mysql.connector import pooling

from config import DB_CONFIG

logger = logging.getLogger(__name__)


_pool: pooling.MySQLConnectionPool | None = None


def init_pool() -> None:
    """Initialise the connection pool. Call once at startup."""
    global _pool
    _pool = pooling.MySQLConnectionPool(**DB_CONFIG)
    logger.info("MySQL connection pool '%s' initialised (size=%d)",
                DB_CONFIG["pool_name"], DB_CONFIG["pool_size"])


@contextmanager
def get_conn():
    """Context manager that yields a pooled connection and commits/rolls back."""
    if _pool is None:
        raise RuntimeError("Call init_pool() before using get_conn()")
    conn = _pool.get_connection()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()   


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS regions (
    id           INT AUTO_INCREMENT PRIMARY KEY,
    name         VARCHAR(100) NOT NULL,
    country_code CHAR(2)      NOT NULL,
    latitude     DECIMAL(8,5) NOT NULL,
    longitude    DECIMAL(8,5) NOT NULL,
    timezone     VARCHAR(50)  NOT NULL DEFAULT 'UTC',
    UNIQUE KEY uq_region (name, country_code)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE IF NOT EXISTS weather_conditions (
    id          INT AUTO_INCREMENT PRIMARY KEY,
    code        VARCHAR(20)  NOT NULL UNIQUE,
    description VARCHAR(100) NOT NULL,
    category    VARCHAR(50)  NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE IF NOT EXISTS weather_readings (
    id               BIGINT AUTO_INCREMENT PRIMARY KEY,
    region_id        INT          NOT NULL,
    condition_id     INT,
    recorded_at      DATETIME     NOT NULL,
    temp_celsius     DECIMAL(5,2),
    feels_like       DECIMAL(5,2),
    humidity_pct     DECIMAL(5,2),
    pressure_hpa     DECIMAL(7,2),
    wind_speed_ms    DECIMAL(6,2),
    wind_dir_deg     SMALLINT,
    precipitation_mm DECIMAL(6,2) DEFAULT 0.00,
    cloud_cover_pct  TINYINT UNSIGNED,
    source_api       VARCHAR(30),
    FOREIGN KEY (region_id)    REFERENCES regions(id),
    FOREIGN KEY (condition_id) REFERENCES weather_conditions(id),
    INDEX idx_region_time (region_id, recorded_at),
    UNIQUE KEY uq_reading (region_id, recorded_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE IF NOT EXISTS model_runs (
    id            INT AUTO_INCREMENT PRIMARY KEY,
    region_id     INT         NOT NULL,
    run_at        DATETIME    NOT NULL DEFAULT CURRENT_TIMESTAMP,
    model_name    VARCHAR(50) NOT NULL,
    training_rows INT,
    mae           DECIMAL(6,3),
    rmse          DECIMAL(6,3),
    horizon_days  TINYINT     DEFAULT 5,
    FOREIGN KEY (region_id) REFERENCES regions(id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE IF NOT EXISTS forecasts (
    id            BIGINT AUTO_INCREMENT PRIMARY KEY,
    region_id     INT          NOT NULL,
    model_run_id  INT          NOT NULL,
    forecast_for  DATETIME     NOT NULL,
    generated_at  DATETIME     NOT NULL DEFAULT CURRENT_TIMESTAMP,
    temp_min      DECIMAL(5,2),
    temp_max      DECIMAL(5,2),
    temp_mean     DECIMAL(5,2),
    precip_prob   DECIMAL(4,3),
    wind_speed_ms DECIMAL(6,2),
    confidence_lo DECIMAL(5,2),
    confidence_hi DECIMAL(5,2),
    FOREIGN KEY (region_id)    REFERENCES regions(id),
    FOREIGN KEY (model_run_id) REFERENCES model_runs(id),
    INDEX idx_forecast_region_time (region_id, forecast_for)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
"""


def bootstrap_schema() -> None:
    """Create all tables if they don't already exist."""
    with get_conn() as conn:
        cursor = conn.cursor()
        for statement in SCHEMA_SQL.strip().split(";"):
            statement = statement.strip()
            if statement:
                cursor.execute(statement)
        cursor.close()
    logger.info("Schema bootstrap complete.")



_region_id_cache:    dict[tuple, int] = {}
_condition_id_cache: dict[str, int]   = {}


def get_or_create_region(conn, name: str, country_code: str,
                          latitude: float, longitude: float,
                          timezone: str) -> int:
    """Return the region PK, inserting the row if it doesn't exist."""
    key = (name, country_code)
    if key in _region_id_cache:
        return _region_id_cache[key]

    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT INTO regions (name, country_code, latitude, longitude, timezone)
        VALUES (%s, %s, %s, %s, %s)
        ON DUPLICATE KEY UPDATE
            latitude  = VALUES(latitude),
            longitude = VALUES(longitude),
            timezone  = VALUES(timezone)
        """,
        (name, country_code, latitude, longitude, timezone),
    )
    cursor.execute(
        "SELECT id FROM regions WHERE name=%s AND country_code=%s", key
    )
    row = cursor.fetchone()
    cursor.close()

    region_id = row[0]
    _region_id_cache[key] = region_id
    return region_id


def get_or_create_condition(conn, code: str, description: str,
                             category: str) -> int:
    """Return the weather_condition PK, inserting if new."""
    if code in _condition_id_cache:
        return _condition_id_cache[code]

    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT INTO weather_conditions (code, description, category)
        VALUES (%s, %s, %s)
        ON DUPLICATE KEY UPDATE
            description = VALUES(description),
            category    = VALUES(category)
        """,
        (code, description, category),
    )
    cursor.execute("SELECT id FROM weather_conditions WHERE code=%s", (code,))
    row = cursor.fetchone()
    cursor.close()

    cond_id = row[0]
    _condition_id_cache[code] = cond_id
    return cond_id


def upsert_reading(conn, reading: dict) -> None:
    """
    Insert a weather reading row, or update all measurement columns if a
    row for (region_id, recorded_at) already exists.

    Expected keys in `reading`:
        region_id, condition_id, recorded_at,
        temp_celsius, feels_like, humidity_pct, pressure_hpa,
        wind_speed_ms, wind_dir_deg, precipitation_mm, cloud_cover_pct,
        source_api
    """
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT INTO weather_readings
            (region_id, condition_id, recorded_at,
             temp_celsius, feels_like, humidity_pct, pressure_hpa,
             wind_speed_ms, wind_dir_deg, precipitation_mm,
             cloud_cover_pct, source_api)
        VALUES
            (%(region_id)s, %(condition_id)s, %(recorded_at)s,
             %(temp_celsius)s, %(feels_like)s, %(humidity_pct)s,
             %(pressure_hpa)s, %(wind_speed_ms)s, %(wind_dir_deg)s,
             %(precipitation_mm)s, %(cloud_cover_pct)s, %(source_api)s)
        ON DUPLICATE KEY UPDATE
            condition_id     = VALUES(condition_id),
            temp_celsius     = VALUES(temp_celsius),
            feels_like       = VALUES(feels_like),
            humidity_pct     = VALUES(humidity_pct),
            pressure_hpa     = VALUES(pressure_hpa),
            wind_speed_ms    = VALUES(wind_speed_ms),
            wind_dir_deg     = VALUES(wind_dir_deg),
            precipitation_mm = VALUES(precipitation_mm),
            cloud_cover_pct  = VALUES(cloud_cover_pct),
            source_api       = VALUES(source_api)
        """,
        reading,
    )
    cursor.close()
    logger.debug("Upserted reading for region_id=%d at %s",
                 reading["region_id"], reading["recorded_at"])
