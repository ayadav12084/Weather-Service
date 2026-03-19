import logging
from flask import Flask, render_template, jsonify, request, redirect, url_for, abort

# Analytics layer (adjust sys.path if not co-located)
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'weather_analytics'))

from db import fetch_region_ids, get_connection
from trend_analysis import chart_payload
from predictor import forecast_payload, run_forecast

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_all_regions() -> list[dict]:
    """Return all regions as a list of dicts with id, name, country_code."""
    with get_connection() as conn:
        with conn.cursor(dictionary=True) as cur:
            cur.execute(
                "SELECT id, name, country_code, latitude, longitude "
                "FROM regions ORDER BY name"
            )
            return cur.fetchall()


def _get_region(region_id: int) -> dict | None:
    with get_connection() as conn:
        with conn.cursor(dictionary=True) as cur:
            cur.execute(
                "SELECT id, name, country_code, latitude, longitude, timezone "
                "FROM regions WHERE id = %s",
                (region_id,)
            )
            return cur.fetchone()


def _latest_reading(region_id: int) -> dict | None:
    sql = """
        SELECT
            wr.recorded_at, wr.temp_celsius, wr.feels_like,
            wr.humidity_pct, wr.pressure_hpa, wr.wind_speed_ms,
            wr.wind_dir_deg, wr.precipitation_mm, wr.cloud_cover_pct,
            wc.description AS condition, wc.category
        FROM weather_readings wr
        LEFT JOIN weather_conditions wc ON wc.id = wr.condition_id
        WHERE wr.region_id = %s
        ORDER BY wr.recorded_at DESC
        LIMIT 1
    """
    with get_connection() as conn:
        with conn.cursor(dictionary=True) as cur:
            cur.execute(sql, (region_id,))
            return cur.fetchone()


# ---------------------------------------------------------------------------
# HTML routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    regions = _get_all_regions()
    if not regions:
        return render_template("error.html", message="No regions configured yet.")
    return redirect(url_for("dashboard", region_id=regions[0]["id"]))


@app.route("/region/<int:region_id>")
def dashboard(region_id: int):
    region = _get_region(region_id)
    if not region:
        abort(404)
    regions = _get_all_regions()
    current = _latest_reading(region_id)
    return render_template(
        "dashboard.html",
        region=region,
        regions=regions,
        current=current,
    )


# ---------------------------------------------------------------------------
# JSON API
# ---------------------------------------------------------------------------

@app.route("/api/regions")
def api_regions():
    return jsonify(_get_all_regions())


@app.route("/api/chart/<int:region_id>")
def api_chart(region_id: int):
    days = request.args.get("days", 7, type=int)
    days = max(1, min(days, 90))   # clamp 1–90
    try:
        data = chart_payload(region_id, days=days)
        return jsonify(data)
    except Exception as exc:
        logger.error("chart_payload error: %s", exc, exc_info=True)
        return jsonify({"error": str(exc)}), 500


@app.route("/api/forecast/<int:region_id>")
def api_forecast(region_id: int):
    horizon = request.args.get("horizon", 5, type=int)
    horizon = max(1, min(horizon, 14))
    try:
        data = forecast_payload(region_id, horizon_days=horizon)
        return jsonify(data)
    except Exception as exc:
        logger.error("forecast_payload error: %s", exc, exc_info=True)
        return jsonify({"error": str(exc)}), 500


@app.route("/api/forecast/run", methods=["POST"])
def api_run_forecast():
    """Trigger a forecast model run. Body: {region_id, model, horizon_days}."""
    body       = request.get_json(force=True) or {}
    region_id  = body.get("region_id")
    model      = body.get("model", "prophet")
    horizon    = body.get("horizon_days", 5)

    if not region_id:
        return jsonify({"error": "region_id required"}), 400

    try:
        result = run_forecast(region_id, horizon_days=horizon, model=model)
        return jsonify(result)
    except Exception as exc:
        logger.error("run_forecast error: %s", exc, exc_info=True)
        return jsonify({"error": str(exc)}), 500


# ---------------------------------------------------------------------------
# Error pages
# ---------------------------------------------------------------------------

@app.errorhandler(404)
def not_found(e):
    return render_template("error.html", message="Page not found."), 404


@app.errorhandler(500)
def server_error(e):
    return render_template("error.html", message="Internal server error."), 500


if __name__ == "__main__":
    app.run(debug=True, port=5000)
