"""Run the pipeline once and exit.

Use it as a Render Cron Job / Background Worker, or locally:

    python -m worker.run_daily
    python -m worker.run_daily --date 2026-10-22 --force
    python -m worker.run_daily --source ecmwf
    python -m worker.run_daily --source gfs --source ecmwf

With no --source it runs every source in settings.sources. The sources are independent: one that
fails does not stop the others, and each adds its own products to the same run manifest. The exit
code is 1 only when none of them finished.
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, timezone

from app.config import settings
from app.core.pipeline import DOWNLOADERS, run_pipeline


def main() -> int:

    parser = argparse.ArgumentParser(description="Daily lightning forecast run")
    parser.add_argument("--date", help="Reference date YYYY-MM-DD (default: today in UTC)")
    parser.add_argument(
        "--force", action="store_true", help="Reprocess even if the run already exists",
    )
    parser.add_argument(
        "--source", action="append", choices=sorted(DOWNLOADERS),
        help="Input model; repeatable (default: settings.sources)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
    )

    log = logging.getLogger("worker")

    reference = None
    if args.date:
        reference = datetime.strptime(args.date, "%Y-%m-%d").replace(tzinfo=timezone.utc)

    sources = args.source or settings.source_list

    manifest = None
    failed: dict[str, str] = {}

    for source in sources:
        log.info("=== %s ===", source)
        try:
            manifest = run_pipeline(
                settings, reference_time=reference, force=args.force, source=source,
            )
        except Exception as error:                              # noqa: BLE001
            # One source failing is a worse forecast, not a failed run: the other one is still
            # worth publishing, and the manifest records which source is missing.
            log.error("%s failed: %s", source, error)
            failed[source] = f"{type(error).__name__}: {error}"

    if manifest is None:
        log.error("No source finished: %s", failed)
        return 1

    print(f"\nRun {manifest['run_id']} finished.")
    for source, entry in manifest.get("sources", {}).items():
        print(f"\n{source}: {entry['source']}")
        print(f"  Regions: {', '.join(entry['regions']['processed'])}")
        for name, product in entry["products"].items():
            print(f"  {name}: {product['files']}")
    for source, error in failed.items():
        print(f"\n{source}: FAILED -- {error}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
