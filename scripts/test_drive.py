"""Check the Google Drive connection and report which models are reachable.

Run this before deploying, so a credential or sharing problem shows up here
instead of inside the daily run.

    # service account + single zip
    export GDRIVE_SERVICE_ACCOUNT_JSON="$(cat service-account.json)"
    export GDRIVE_ASSETS_ZIP_ID=1AbC...
    python scripts/test_drive.py

    # service account + mirrored folders
    export GDRIVE_CLASSIFICATION_FOLDER_ID=1AbC...
    export GDRIVE_REGRESSOR_FOLDER_ID=1DeF...
    python scripts/test_drive.py

    # publicly shared items, no credential
    export GDRIVE_USE_GDOWN=true
    python scripts/test_drive.py

Pass --download to actually fetch everything into DATA_DIR/assets; without
it the script only inspects metadata, which takes a couple of seconds.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import Settings  # noqa: E402
from app.core import drive  # noqa: E402


def describe_source(cfg: Settings) -> str:
    if cfg.assets_local_dir:
        return f"local directory ({cfg.assets_local_dir})"
    if cfg.gdrive_assets_zip_id:
        return f"single zip on Drive (id {cfg.gdrive_assets_zip_id})"
    if cfg.gdrive_classification_folder_id or cfg.gdrive_regressor_folder_id:
        return "mirrored Drive folders"
    return "NOT CONFIGURED"


def inspect_metadata(cfg: Settings) -> int:
    """Query Drive without downloading, to validate access."""

    service = drive._build_service(cfg)
    print("Credential accepted: the Drive API client was created.\n")

    if cfg.gdrive_assets_zip_id:
        meta = (
            service.files()
            .get(
                fileId=cfg.gdrive_assets_zip_id,
                fields="id, name, size, mimeType, modifiedTime",
                supportsAllDrives=True,
            )
            .execute()
        )
        size = int(meta.get("size", 0)) / 1e6
        print(f"  zip: {meta['name']}  ({size:.1f} MB)")
        print(f"       type {meta['mimeType']}, modified {meta['modifiedTime']}")
        return 0

    folders = [
        ("classification", cfg.gdrive_classification_folder_id),
        ("regressor", cfg.gdrive_regressor_folder_id),
        ("shapefiles", cfg.gdrive_shapefiles_folder_id),
    ]

    for label, folder_id in folders:
        if not folder_id:
            continue
        children = drive._list_children(service, folder_id)
        subfolders = [c for c in children if c["mimeType"] == drive.FOLDER_MIME]
        print(
            f"  {label}: {len(children)} entries "
            f"({len(subfolders)} subfolders) in folder {folder_id}"
        )
        for child in children[:8]:
            kind = "dir " if child["mimeType"] == drive.FOLDER_MIME else "file"
            print(f"       [{kind}] {child['name']}")
        if len(children) > 8:
            print(f"       ... and {len(children) - 8} more")

    return 0


def main() -> int:

    parser = argparse.ArgumentParser(description="Test the Google Drive connection")
    parser.add_argument(
        "--download", action="store_true",
        help="Actually download the models into DATA_DIR/assets",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(message)s")

    cfg = Settings()

    print(f"\nSource:     {describe_source(cfg)}")
    print(f"Assets dir: {cfg.assets_dir}")
    print(f"Auth:       {'gdown (public items)' if cfg.gdrive_use_gdown else 'service account'}")
    print(f"Regions:    {len(cfg.region_list)}")
    print("-" * 62)

    if describe_source(cfg) == "NOT CONFIGURED":
        print(
            "\nNothing to test. Set GDRIVE_ASSETS_ZIP_ID, or "
            "GDRIVE_CLASSIFICATION_FOLDER_ID + GDRIVE_REGRESSOR_FOLDER_ID, "
            "or ASSETS_LOCAL_DIR."
        )
        return 1

    if args.download:
        drive.ensure_assets(cfg, force=True)
    elif not cfg.gdrive_use_gdown and not cfg.assets_local_dir:
        try:
            inspect_metadata(cfg)
        except Exception as error:                              # noqa: BLE001
            print(f"\nDrive access FAILED: {type(error).__name__}: {error}")
            print(
                "\nMost common causes:\n"
                "  - the folder/file was not shared with the service account email\n"
                "  - the Google Drive API is not enabled for the project\n"
                "  - the pasted JSON is truncated or is not a service account key"
            )
            return 1
    else:
        print("Metadata inspection needs a service account; use --download instead.")

    print("\n" + "-" * 62)
    missing = drive.missing_assets(cfg, cfg.region_list)
    complete = [r for r in cfg.region_list if r not in missing]

    print(f"Regions with a complete model set on disk: {len(complete)}/{len(cfg.region_list)}")
    for region in complete:
        print(f"  [ok]      {region}")
    for region, files in missing.items():
        print(f"  [missing] {region}  ({len(files)} file(s), e.g. {Path(files[0]).name})")

    if not complete and not args.download:
        print("\nNothing on disk yet. Re-run with --download to fetch the models.")

    return 0 if (complete or not args.download) else 1


if __name__ == "__main__":
    sys.exit(main())
