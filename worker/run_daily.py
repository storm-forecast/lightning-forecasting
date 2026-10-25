"""Run the pipeline once and exit.

Use it as a Render Cron Job / Background Worker, or locally:

    python -m worker.run_daily
    python -m worker.run_daily --date 2026-10-22 --force
    python -m worker.run_daily --source ecmwf
    python -m worker.run_daily --source gfs --source ecmwf

With no --source it runs every source in settings.sources. The sources are independent: one that
fails does not stop the others, and each adds its own products to the same run manifest. The exit
code is 1 only when none of them finished.

With more than one source the run happens in three phases, so that both publish the same valid
times -- a page stepping through both frame by frame breaks otherwise:

    1. download and compute parameters for every source, and intersect the valid times
    2. run each source restricted to that intersection
    3. trim the lightning products to the times they share, for the all-null drop inside the
       prediction, which no amount of input alignment can anticipate

Phase 1 is why a source that cannot deliver its data fails before any model is loaded. Pass
--no-align to skip all of it and let each source run on whatever it has.
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, timezone

from app.config import settings
from app.core import state
from app.core.pipeline import (
    DOWNLOADERS,
    align_lightning_times,
    common_times,
    prepare_inputs,
    run_pipeline,
)


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
    parser.add_argument(
        "--no-align", action="store_true",
        help="Let each source run on its own valid times instead of the shared ones",
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

    # ---- phase 1: what each source can offer, and what they share
    shared = None

    if len(sources) > 1 and not args.no_align:
        available: dict[str, list] = {}
        for source in sources:
            log.info("=== %s: entrada ===", source)
            try:
                available[source] = prepare_inputs(settings, reference, source)
            except Exception as error:                          # noqa: BLE001
                # Cheaper here than after the models are loaded, and the other source carries on.
                log.error("%s failed while preparing its input: %s", source, error)
                failed[source] = f"{type(error).__name__}: {error}"
        sources = [s for s in sources if s in available]
        shared = common_times(available)
        if shared is not None:
            log.info(
                "Shared period: %d valid times, %s to %s",
                len(shared), shared[0], shared[-1],
            )

    # ---- phase 2: the run itself, on the shared period
    for index, source in enumerate(sources):
        log.info("=== %s ===", source)
        try:
            manifest = run_pipeline(
                settings, reference_time=reference, force=args.force, source=source,
                restrict_times=shared,
                # Keep the downloads until the last source has used them -- without overriding a
                # configuration that asked for the work directory to be left alone.
                cleanup=settings.cleanup_workdir and index == len(sources) - 1,
            )
        except Exception as error:                              # noqa: BLE001
            # One source failing is a worse forecast, not a failed run: the other one is still
            # worth publishing, and the manifest records which source is missing.
            log.error("%s failed: %s", source, error)
            failed[source] = f"{type(error).__name__}: {error}"

    if manifest is None:
        log.error("No source finished: %s", failed)
        return 1

    # ---- phase 3: the net under the prediction's own dropping of valid times
    if not args.no_align:
        aligned = align_lightning_times(settings, manifest["run_id"])
        if aligned:
            manifest = state.get_manifest(settings.products_dir, manifest["run_id"]) or manifest
            log.info("Lightning products share %d valid times", len(aligned))

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
