import logging
import logging.handlers
import signal
import sys
import time

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.events import EVENT_JOB_ERROR, EVENT_JOB_EXECUTED

import db
import fetcher
from config import (
    DB_CONFIG,
    FETCH_INTERVAL_MINUTES,
    LOG_FILE,
    LOG_LEVEL,
    REGIONS,
    SCHEDULER_TIMEZONE,
)



def setup_logging() -> None:
    level = getattr(logging, LOG_LEVEL.upper(), logging.INFO)
    fmt   = "%(asctime)s [%(levelname)s] %(name)s — %(message)s"

    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    try:
        file_handler = logging.handlers.RotatingFileHandler(
            LOG_FILE, maxBytes=10 * 1024 * 1024, backupCount=5
        )
        file_handler.setFormatter(logging.Formatter(fmt))
        handlers.append(file_handler)
    except OSError as exc:
        # If the log path isn't writable (e.g. in dev), just use stdout
        print(f"Warning: cannot open log file {LOG_FILE}: {exc}")
/
logging.basicConfig(level=level, format=fmt, handlers=handlers)


logger = logging.getLogger(__name__)




def fetch_all_regions() -> None:
    logger.info("=== Starting fetch cycle for %d region(s) ===", len(REGIONS))
    success = 0
    failure = 0
    for region in REGIONS:
        ok = fetcher.fetch_and_store(region)
        if ok:
            success += 1
        else:
            failure += 1
    logger.info(
        "=== Fetch cycle complete — %d succeeded, %d failed ===",
        success, failure,
    )



def _on_job_error(event) -> None:
    logger.error("Scheduler job raised an unhandled exception: %s",
                 event.exception, exc_info=event.traceback)


def _on_job_executed(event) -> None:
    logger.debug("Job '%s' ran in %.2fs", event.job_id, event.retval or 0)



def main() -> None:
    setup_logging()
    logger.info("Weather ingestion daemon starting …")

    db.init_pool()
    db.bootstrap_schema()

    logger.info("Running initial fetch …")
    fetch_all_regions()

    scheduler = BackgroundScheduler(timezone=SCHEDULER_TIMEZONE)
    scheduler.add_job(
        fetch_all_regions,
        trigger="interval",
        minutes=FETCH_INTERVAL_MINUTES,
        id="weather_fetch",
        max_instances=1,         
        coalesce=True,           
    )
    scheduler.add_listener(_on_job_error,    EVENT_JOB_ERROR)
    scheduler.add_listener(_on_job_executed, EVENT_JOB_EXECUTED)
    scheduler.start()

    logger.info(
        "Scheduler started — fetching every %d minutes. Press Ctrl+C to stop.",
        FETCH_INTERVAL_MINUTES,
    )

    # Graceful shutdown on SIGINT / SIGTERM
    def _shutdown(signum, frame):
        logger.info("Signal %d received — shutting down scheduler …", signum)
        scheduler.shutdown(wait=False)
        sys.exit(0)

    signal.signal(signal.SIGINT,  _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    # Keep the main thread alive
    while True:
        time.sleep(60)


if __name__ == "__main__":
    main()


