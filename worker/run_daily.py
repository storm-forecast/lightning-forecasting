"""Run the pipeline once and exit.

Use it as a Render Cron Job / Background Worker, or locally:

    python -m worker.run_daily
    python -m worker.run_daily --date 2026-10-22 --force
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, timezone

from app.config import settings
from app.core.pipeline import run_pipeline


def main() -> int:

    parser = argparse.ArgumentParser(description="Daily lightning forecast run")
    parser.add_argument("--date", help="Reference date YYYY-MM-DD (default: today in UTC)")
    parser.add_argument(
        "--force", action="store_true", help="Reprocess even if the run already exists",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
    )

    reference = None
    if args.date:
        reference = datetime.strptime(args.date, "%Y-%m-%d").replace(tzinfo=timezone.utc)

    try:
        manifest = run_pipeline(settings, reference_time=reference, force=args.force)
    except Exception as error:                                  # noqa: BLE001
        logging.getLogger("worker").error("Run failed: %s", error)
        return 1

    print(f"\nRun {manifest['run_id']} finished.")
    print(f"Regions: {', '.join(manifest['regions']['processed'])}")
    for name, product in manifest["products"].items():
        print(f"  {name}: {product['files']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
