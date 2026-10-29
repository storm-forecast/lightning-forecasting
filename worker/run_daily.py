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
import json
import logging
import os
import sys
import traceback
from datetime import datetime, timezone

from app.config import settings
from app.core import state
from app.core.pipeline import (
    DOWNLOADERS,
    align_lightning_times,
    common_times,
    prepare_inputs,
    run_pipeline,
    run_precipitation_only,
)


def relatar(failed: dict[str, str], tracos: dict[str, str], pedidas: list[str]) -> None:
    """Make a source's failure visible where somebody is already looking.

    A source failing is not a failed run -- the other one is still worth publishing, and that is the
    right behaviour. The cost of it is that a permanently broken source looks like a quiet success for
    as long as nobody opens the log: the run is green, the release appears, and two of the four files
    are simply never there. That is how the ECMWF side went missing for days without anyone noticing.

    So the failure is written where it cannot be missed: an ::error:: annotation, which GitHub lists at
    the top of the run page next to the warnings, and the job summary, which is the first thing the page
    shows. Neither fails the run.
    """
    if not failed:
        return
    for source, erro in failed.items():
        # One line, no newlines: an annotation stops at the first break.
        print(f"::error title=Fonte {source} falhou::{erro}".replace("\n", " "), flush=True)

    # Left where the workflow can pick it up and publish it beside the data, so the outcome is readable
    # without access to this repository. The type and the message, not the traceback: the traceback
    # carries the runner's paths and belongs in the run page, which is private.
    destino = os.environ.get("FALHAS_JSON")
    if destino:
        try:
            with open(destino, "w", encoding="utf-8") as saida:
                json.dump(failed, saida, ensure_ascii=False)
        except OSError as erro:
            log.warning("could not write %s: %s", destino, erro)

    resumo = os.environ.get("GITHUB_STEP_SUMMARY")
    if not resumo:
        return
    with open(resumo, "a", encoding="utf-8") as saida:
        saida.write("## Fontes que falharam\n\n")
        saida.write(f"Pedidas: `{', '.join(pedidas) or '(nenhuma)'}`\n\n")
        for source, erro in failed.items():
            saida.write(f"### {source}\n\n```\n{erro}\n```\n\n")
            traco = tracos.get(source)
            if traco:
                saida.write(f"<details><summary>Traceback</summary>\n\n```\n{traco}```\n\n</details>\n\n")


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
    parser.add_argument(
        "--rain-only", action="store_true",
        help="Generate only worldwide precipitation; skips lightning models",
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
    # Kept because `sources` is filtered below to the ones that could offer their input: the report
    # should say what was asked for, not what survived.
    pedidas = list(sources)

    manifest = None
    failed: dict[str, str] = {}
    tracos: dict[str, str] = {}

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
                log.error("%s failed while preparing its input: %s", source, error, exc_info=True)
                failed[source] = f"{type(error).__name__}: {error}"
                tracos[source] = traceback.format_exc()
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
            if args.rain_only:
                manifest = run_precipitation_only(
                    settings, reference_time=reference, force=args.force, source=source,
                    cleanup=settings.cleanup_workdir and index == len(sources) - 1,
                )
            else:
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
            log.error("%s failed: %s", source, error, exc_info=True)
            failed[source] = f"{type(error).__name__}: {error}"
            tracos[source] = traceback.format_exc()

    if manifest is None:
        log.error("No source finished: %s", failed)
        relatar(failed, tracos, pedidas)
        return 1

    # ---- phase 3: the net under the prediction's own dropping of valid times
    if not args.no_align and not args.rain_only:
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
    relatar(failed, tracos, pedidas)
    return 0


if __name__ == "__main__":
    sys.exit(main())
