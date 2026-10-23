"""Fetches the trained models that live on Google Drive.

Three possible sources, in priority order:

1. ASSETS_LOCAL_DIR
   The model tree is already on disk (useful in development, or when the
   models ship inside the image).

2. GDRIVE_ASSETS_ZIP_ID
   A single .zip on Drive containing Model-Trained/, Model-Trained-new/ and,
   optionally, shapefiles_clusters_with_buffer/. This is the simplest and
   fastest option.

3. GDRIVE_CLASSIFICATION_FOLDER_ID + GDRIVE_REGRESSOR_FOLDER_ID
   Recursive folder mirror through the Drive API.

Authentication: a service account (recommended, works with private folders)
or gdown (publicly shared items only).

Expected layout under ASSETS_DIR once the download finishes:

    Model-Trained/<region>/results/train_data/classification/
        nw_classification_model_<region>_6H.h5
        nw_mean_<region>_6H.npy
        nw_std_<region>_6H.npy
    Model-Trained-new/<region>/results/train_data/regressor/
        regressor_model_<region>_6H.keras
        mean_regressor_<region>_6H.npy
        std_regressor_<region>_6H.npy
        mean_data_regressor_<region>_6H.npy
        std_data_regressor_<region>_6H.npy
    shapefiles_clusters_with_buffer/<region>.shp        (optional)
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import zipfile
from pathlib import Path

from app.config import Settings

log = logging.getLogger(__name__)

SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]
FOLDER_MIME = "application/vnd.google-apps.folder"
MANIFEST_NAME = ".drive_manifest.json"


# ============================================================
# Expected model paths
# ============================================================


def classification_paths(settings: Settings, region: str) -> dict[str, Path]:
    base = (
        settings.classification_root / region / "results" / "train_data" / "classification"
    )
    steps = settings.steps
    return {
        "model": base / f"nw_classification_model_{region}_{steps}.h5",
        "mean": base / f"nw_mean_{region}_{steps}.npy",
        "std": base / f"nw_std_{region}_{steps}.npy",
    }


def regressor_paths(settings: Settings, region: str) -> dict[str, Path]:
    base = settings.regressor_root / region / "results" / "train_data" / "regressor"
    steps = settings.steps
    return {
        "model": base / f"regressor_model_{region}_{steps}.keras",
        "mean_label": base / f"mean_regressor_{region}_{steps}.npy",
        "std_label": base / f"std_regressor_{region}_{steps}.npy",
        "mean_data": base / f"mean_data_regressor_{region}_{steps}.npy",
        "std_data": base / f"std_data_regressor_{region}_{steps}.npy",
    }


def missing_assets(settings: Settings, regions: list[str]) -> dict[str, list[str]]:
    """Maps region -> list of model files that are missing."""

    report: dict[str, list[str]] = {}

    for region in regions:
        missing = [
            str(path)
            for group in (classification_paths(settings, region),
                          regressor_paths(settings, region))
            for path in group.values()
            if not path.exists()
        ]
        if missing:
            report[region] = missing

    return report


# ============================================================
# Drive API (service account)
# ============================================================


def _build_service(settings: Settings):

    from google.oauth2 import service_account
    from googleapiclient.discovery import build

    info = None

    if settings.gdrive_service_account_json.strip():
        raw = settings.gdrive_service_account_json.strip()
        # Accepts either the JSON pasted inline or a path to a file.
        if raw.startswith("{"):
            info = json.loads(raw)
        elif Path(raw).exists():
            info = json.loads(Path(raw).read_text(encoding="utf-8"))

    if info is None and settings.gdrive_service_account_file.strip():
        info = json.loads(
            Path(settings.gdrive_service_account_file).read_text(encoding="utf-8")
        )

    if info is None:
        raise RuntimeError(
            "Missing Drive credential. Set GDRIVE_SERVICE_ACCOUNT_JSON (the "
            "service account JSON) or set GDRIVE_USE_GDOWN=true for publicly "
            "shared items."
        )

    credentials = service_account.Credentials.from_service_account_info(
        info, scopes=SCOPES,
    )
    return build("drive", "v3", credentials=credentials, cache_discovery=False)


def _list_children(service, folder_id: str) -> list[dict]:

    items: list[dict] = []
    page_token = None

    while True:
        response = (
            service.files()
            .list(
                q=f"'{folder_id}' in parents and trashed=false",
                fields="nextPageToken, files(id, name, mimeType, md5Checksum, size)",
                pageSize=1000,
                supportsAllDrives=True,
                includeItemsFromAllDrives=True,
                pageToken=page_token,
            )
            .execute()
        )
        items.extend(response.get("files", []))
        page_token = response.get("nextPageToken")
        if not page_token:
            break

    return items


def _download_file(service, file_id: str, destination: Path) -> None:

    from googleapiclient.http import MediaIoBaseDownload

    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp = destination.with_suffix(destination.suffix + ".part")

    request = service.files().get_media(fileId=file_id, supportsAllDrives=True)

    with open(tmp, "wb") as handle:
        downloader = MediaIoBaseDownload(handle, request, chunksize=8 * 1024 * 1024)
        done = False
        while not done:
            _status, done = downloader.next_chunk()

    os.replace(tmp, destination)


def _mirror_folder(
    service,
    folder_id: str,
    destination: Path,
    manifest: dict,
    prefix: str = "",
) -> None:
    """Recursively mirror a Drive folder into destination."""

    destination.mkdir(parents=True, exist_ok=True)

    for item in _list_children(service, folder_id):

        name = item["name"]
        relative = f"{prefix}/{name}" if prefix else name

        if item["mimeType"] == FOLDER_MIME:
            _mirror_folder(
                service, item["id"], destination / name, manifest, relative,
            )
            continue

        local = destination / name
        signature = item.get("md5Checksum") or item.get("size")
        cached = manifest.get(relative)

        if local.exists() and cached is not None and cached == signature:
            continue

        log.info("Drive -> %s", relative)
        _download_file(service, item["id"], local)
        manifest[relative] = signature


# ============================================================
# gdown (public items)
# ============================================================


def _gdown_folder(folder_id: str, destination: Path) -> None:
    import gdown

    destination.mkdir(parents=True, exist_ok=True)
    gdown.download_folder(
        id=folder_id, output=str(destination), quiet=False, use_cookies=False,
    )


def _gdown_file(file_id: str, destination: Path) -> None:
    import gdown

    destination.parent.mkdir(parents=True, exist_ok=True)
    gdown.download(id=file_id, output=str(destination), quiet=False)


# ============================================================
# Zip
# ============================================================


def _safe_extract(zip_path: Path, destination: Path) -> None:
    """Extract the zip, skipping members with absolute paths or '..'."""

    destination.mkdir(parents=True, exist_ok=True)
    root = destination.resolve()

    with zipfile.ZipFile(zip_path) as archive:
        for member in archive.infolist():
            if member.is_dir():
                continue
            target = (root / member.filename).resolve()
            if not str(target).startswith(str(root) + os.sep):
                log.warning("Skipping member with suspicious path: %s", member.filename)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(member) as src, open(target, "wb") as dst:
                shutil.copyfileobj(src, dst, length=4 * 1024 * 1024)


def _flatten_single_root(destination: Path) -> None:
    """If the zip had a single root folder, lift its contents up one level."""

    entries = [p for p in destination.iterdir() if not p.name.startswith(".")]

    if len(entries) != 1 or not entries[0].is_dir():
        return
    if (destination / "Model-Trained").exists():
        return

    inner = entries[0]
    if not (inner / "Model-Trained").exists():
        return

    log.info("Flattening the zip's root folder: %s", inner.name)
    for item in list(inner.iterdir()):
        shutil.move(str(item), str(destination / item.name))
    inner.rmdir()


# ============================================================
# Public entry point
# ============================================================


def ensure_assets(settings: Settings, force: bool = False) -> Path:
    """Make sure the models are available under settings.assets_dir."""

    assets = settings.assets_dir
    assets.mkdir(parents=True, exist_ok=True)

    regions = settings.region_list

    if settings.assets_local_dir:
        log.info("Using local models in %s", assets)
        return assets

    if not force and not missing_assets(settings, regions):
        log.info("Models already present in %s (disk cache).", assets)
        return assets

    manifest_path = assets / MANIFEST_NAME
    manifest: dict = {}
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            manifest = {}

    # ---------- 1. Single zip ----------
    if settings.gdrive_assets_zip_id:

        zip_path = assets / "_assets.zip"
        log.info("Downloading the model zip from Drive (%s)...", settings.gdrive_assets_zip_id)

        if settings.gdrive_use_gdown:
            _gdown_file(settings.gdrive_assets_zip_id, zip_path)
        else:
            service = _build_service(settings)
            _download_file(service, settings.gdrive_assets_zip_id, zip_path)

        log.info("Extracting %.1f MB...", zip_path.stat().st_size / 1e6)
        _safe_extract(zip_path, assets)
        _flatten_single_root(assets)
        zip_path.unlink(missing_ok=True)

    # ---------- 2. Mirrored folders ----------
    elif settings.gdrive_classification_folder_id or settings.gdrive_regressor_folder_id:

        jobs = [
            (settings.gdrive_classification_folder_id, settings.classification_root),
            (settings.gdrive_regressor_folder_id, settings.regressor_root),
            (settings.gdrive_shapefiles_folder_id, settings.shapefiles_root),
        ]

        if settings.gdrive_use_gdown:
            for folder_id, target in jobs:
                if folder_id:
                    log.info("gdown -> %s", target)
                    _gdown_folder(folder_id, target)
        else:
            service = _build_service(settings)
            for folder_id, target in jobs:
                if folder_id:
                    log.info("Mirroring Drive folder into %s", target)
                    _mirror_folder(service, folder_id, target, manifest, target.name)

    else:
        raise RuntimeError(
            "No model source configured. Set GDRIVE_ASSETS_ZIP_ID, or "
            "GDRIVE_CLASSIFICATION_FOLDER_ID + GDRIVE_REGRESSOR_FOLDER_ID, "
            "or ASSETS_LOCAL_DIR."
        )

    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    report = missing_assets(settings, regions)
    if report:
        for region, files in report.items():
            log.warning(
                "Region %s is incomplete - %d file(s) missing, e.g. %s",
                region, len(files), Path(files[0]).name,
            )

    available = [r for r in regions if r not in report]
    log.info(
        "Models ready: %d of %d regions (%s)",
        len(available), len(regions), ", ".join(available) or "none",
    )

    if not available:
        raise RuntimeError(
            "No region has a complete set of models. Check the folder layout "
            "documented in app/core/drive.py."
        )

    return assets
