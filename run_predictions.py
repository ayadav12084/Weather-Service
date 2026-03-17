
import argparse
import logging
import sys

from config import REGIONS
from db import fetch_region_ids
from predictor import run_forecast

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("run_predictions")

AVAILABLE_MODELS = ["prophet", "random_forest"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run weather forecast models")
    parser.add_argument(
        "--model", default="prophet",
        choices=AVAILABLE_MODELS + ["all"],
        help="Model to run (default: prophet)",
    )
    parser.add_argument(
        "--region", default=None,
        help="Region name to target (default: all configured regions)",
    )
    parser.add_argument(
        "--horizon", type=int, default=5,
        help="Forecast horizon in days (default: 5)",
    )
    args = parser.parse_args()

    models   = AVAILABLE_MODELS if args.model == "all" else [args.model]
    all_ids  = fetch_region_ids()

    if args.region:
        if args.region not in all_ids:
            logger.error("Region '%s' not found in DB. Available: %s",
                         args.region, list(all_ids.keys()))
            sys.exit(1)
        target_regions = {args.region: all_ids[args.region]}
    else:
        target_regions = all_ids

    errors = 0
    for region_name, region_id in target_regions.items():
        for model in models:
            logger.info("── Forecasting  region=%-12s  model=%s", region_name, model)
            try:
                result = run_forecast(region_id, horizon_days=args.horizon, model=model)
                logger.info(
                    "   ✓  model_run_id=%-4d  MAE=%.3f°C  RMSE=%.3f°C  rows=%d",
                    result["model_run_id"], result["mae"],
                    result["rmse"],         result["rows_written"],
                )
            except Exception as exc:
                logger.error("   ✗  %s / %s failed: %s", region_name, model, exc, exc_info=True)
                errors += 1

    if errors:
        logger.warning("Finished with %d error(s)", errors)
        sys.exit(1)
    else:
        logger.info("All forecasts completed successfully")


if __name__ == "__main__":
    main()
