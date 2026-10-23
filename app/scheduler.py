"""Daily run scheduler, embedded in the web service.

On Render, Cron Jobs cannot mount a persistent disk, so the daily run lives
inside the Web Service itself, on a background thread, writing to the disk
mounted at /var/data.
"""

from __future__ import annotations

import logging
import threading
from datetime import datetime, timezone

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger

from app.config import Settings
from app.core import pipeline, state

log = logging.getLogger(__name__)

_scheduler: BackgroundScheduler | None = None


def _job(cfg: Settings) -> None:
    try:
        pipeline.run_pipeline(cfg)
    except pipeline.PipelineBusy:
        log.warning("Scheduled run skipped: another one is already in progress.")
    except Exception:                                           # noqa: BLE001
        log.exception("Scheduled run failed")


def _startup_check(cfg: Settings) -> None:
    """Run the pipeline when today's cycle is missing but already published."""

    run_id = state.run_id_for(datetime.now(timezone.utc))

    if state.get_manifest(cfg.products_dir, run_id) is not None:
        log.info("Run %s is already on disk; nothing to do at startup.", run_id)
        return

    log.info("Run %s missing; checking availability in the NOAA bucket.", run_id)

    from app.core.gfs import GFSDownloader

    downloader = GFSDownloader(
        output_base=cfg.work_dir / "gfs", horizon_hours=cfg.horizon_hours,
    )

    if not downloader.today_run_is_ready():
        log.info(
            "Today's 00z GFS is still incomplete in the bucket; "
            "waiting for the scheduled time (%s UTC).", cfg.run_at_utc,
        )
        return

    _job(cfg)


def start_scheduler(cfg: Settings) -> BackgroundScheduler | None:

    global _scheduler

    if not cfg.enable_scheduler:
        log.info("Scheduler disabled (ENABLE_SCHEDULER=false).")
        return None

    if _scheduler is not None:
        return _scheduler

    hour, minute = (int(part) for part in cfg.run_at_utc.split(":"))

    _scheduler = BackgroundScheduler(timezone="UTC")
    _scheduler.add_job(
        _job,
        trigger=CronTrigger(hour=hour, minute=minute, timezone="UTC"),
        args=[cfg],
        id="daily-run",
        name="Daily lightning forecast",
        max_instances=1,
        coalesce=True,
        misfire_grace_time=3 * 3600,
    )
    _scheduler.start()

    log.info("Scheduler active: every day at %s UTC.", cfg.run_at_utc)

    if cfg.run_on_startup:
        threading.Thread(
            target=_startup_check, args=(cfg,), name="startup-check", daemon=True,
        ).start()

    return _scheduler


def stop_scheduler() -> None:
    global _scheduler
    if _scheduler is not None:
        _scheduler.shutdown(wait=False)
        _scheduler = None
