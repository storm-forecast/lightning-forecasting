"""Routes of the lightning prediction API."""

from __future__ import annotations

import logging
import threading
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, status
from fastapi.responses import FileResponse, JSONResponse

from app.api.deps import require_api_key, resolve_run, settings_dep
from app.config import Settings
from app.core import pipeline, state
from app.core.regions import REGION_META, VAR_DETER, VAR_PROB

log = logging.getLogger(__name__)

router = APIRouter()


# ============================================================
# Metadata and status
# ============================================================


@router.get("/health", summary="Health check")
def health() -> dict[str, Any]:
    return {"status": "ok", "time": state.utcnow_iso()}


@router.get("/status", summary="Status of the last run")
def get_status(cfg: Settings = Depends(settings_dep)) -> dict[str, Any]:

    run_state = state.RunState(cfg.state_file).read()
    latest = state.get_latest_run_id(cfg.products_dir)

    return {
        "run_state": run_state,
        "running_now": pipeline.pipeline_is_running(),
        "latest_run": latest,
        "available_runs": state.list_runs(cfg.products_dir),
        "scheduler": {
            "enabled": cfg.enable_scheduler,
            "run_at_utc": cfg.run_at_utc,
            "next_run": pipeline.next_run_datetime(cfg).isoformat(),
        },
    }


@router.get("/regions", summary="Regions (clusters) and their labels")
def get_regions(cfg: Settings = Depends(settings_dep)) -> dict[str, Any]:
    return {
        "configured": cfg.region_list,
        "catalog": [
            {
                "name": meta.name,
                "label": meta.label,
                "title": meta.title,
                "color": meta.color,
            }
            for meta in REGION_META.values()
        ],
    }


@router.get("/runs", summary="Runs available on disk")
def get_runs(cfg: Settings = Depends(settings_dep)) -> dict[str, Any]:
    return {
        "latest": state.get_latest_run_id(cfg.products_dir),
        "runs": state.list_runs(cfg.products_dir),
    }


@router.get("/manifest", summary="Manifest of a run")
def get_manifest(
    run_id: str = Depends(resolve_run),
    cfg: Settings = Depends(settings_dep),
) -> dict[str, Any]:
    manifest = state.get_manifest(cfg.products_dir, run_id)
    if manifest is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Manifest not found.")
    return manifest


# ============================================================
# Products: lightning
# ============================================================


@router.get("/lightning", summary="Lightning forecast metadata")
def get_lightning(
    run_id: str = Depends(resolve_run),
    cfg: Settings = Depends(settings_dep),
) -> dict[str, Any]:

    manifest = state.get_manifest(cfg.products_dir, run_id) or {}
    product = manifest.get("products", {}).get("lightning")

    if product is None:
        raise HTTPException(
            status.HTTP_404_NOT_FOUND, "This run has no lightning product.",
        )

    return {
        "run_id": run_id,
        "run_time": manifest.get("run_time"),
        "variables": {
            VAR_PROB: "probability of lightning in 6 h (0 to 1)",
            VAR_DETER: "predicted flash count in 6 h",
        },
        **product,
        "download": {
            key: f"/api/v1/download/{run_id}/{value}"
            for key, value in product.get("files", {}).items()
            if isinstance(value, str)
        },
    }


# ============================================================
# Products: precipitation
# ============================================================


@router.get("/precipitation", summary="Precipitation metadata")
def get_precipitation(
    run_id: str = Depends(resolve_run),
    cfg: Settings = Depends(settings_dep),
) -> dict[str, Any]:

    manifest = state.get_manifest(cfg.products_dir, run_id) or {}
    product = manifest.get("products", {}).get("precipitation")

    if product is None:
        raise HTTPException(
            status.HTTP_404_NOT_FOUND,
            "This run has no precipitation product (the GFS carried no 'tp').",
        )

    return {
        "run_id": run_id,
        "run_time": manifest.get("run_time"),
        "variables": {
            "precip_6h": "6 h accumulation (mm)",
            "precip_24h": "24 h accumulation (mm)",
            "precip_total": "accumulation over the whole forecast window (mm)",
        },
        **product,
        "download": {
            key: f"/api/v1/download/{run_id}/{value}"
            for key, value in product.get("files", {}).items()
            if isinstance(value, str)
        },
    }


# ============================================================
# File download
# ============================================================


@router.get("/download/{run_id}/{path:path}", summary="Download a file from the run")
def download(
    run_id: str,
    path: str,
    cfg: Settings = Depends(settings_dep),
) -> FileResponse:

    try:
        target = state.resolve_product(cfg.products_dir, run_id, path)
    except FileNotFoundError:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"File not found: {path}")
    except ValueError:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Invalid path.")

    return FileResponse(
        target,
        filename=target.name,
        media_type="application/x-netcdf",
        headers={"Cache-Control": "public, max-age=3600"},
    )


# ============================================================
# Administration
# ============================================================


@router.post(
    "/admin/run",
    summary="Trigger the run manually",
    dependencies=[Depends(require_api_key)],
)
def trigger_run(
    force: bool = Query(False, description="Reprocess even if the run already exists"),
    date: str | None = Query(
        None, description="Reference date YYYY-MM-DD (default: today in UTC)",
    ),
    cfg: Settings = Depends(settings_dep),
) -> JSONResponse:

    if pipeline.pipeline_is_running():
        raise HTTPException(status.HTTP_409_CONFLICT, "A run is already in progress.")

    reference = None
    if date:
        try:
            reference = datetime.strptime(date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        except ValueError:
            raise HTTPException(status.HTTP_400_BAD_REQUEST, "Invalid date (use YYYY-MM-DD).")

    def task() -> None:
        try:
            pipeline.run_pipeline(cfg, reference_time=reference, force=force)
        except Exception:                                       # noqa: BLE001
            log.exception("Manually triggered run failed")

    threading.Thread(target=task, name="pipeline-manual", daemon=True).start()

    return JSONResponse(
        status_code=status.HTTP_202_ACCEPTED,
        content={
            "accepted": True,
            "message": "Run started in the background; follow it at /api/v1/status.",
            "force": force,
            "reference_date": date,
        },
    )


@router.post(
    "/admin/refresh-models",
    summary="Re-download the models from Google Drive",
    dependencies=[Depends(require_api_key)],
)
def refresh_models(cfg: Settings = Depends(settings_dep)) -> dict[str, Any]:

    from app.core import drive

    drive.ensure_assets(cfg, force=True)
    missing = drive.missing_assets(cfg, cfg.region_list)

    return {
        "assets_dir": str(cfg.assets_dir),
        "complete_regions": [r for r in cfg.region_list if r not in missing],
        "incomplete_regions": {k: len(v) for k, v in missing.items()},
    }


@router.get(
    "/admin/models",
    summary="Status of the models on disk",
    dependencies=[Depends(require_api_key)],
)
def models_status(cfg: Settings = Depends(settings_dep)) -> dict[str, Any]:

    from app.core import drive

    missing = drive.missing_assets(cfg, cfg.region_list)

    return {
        "assets_dir": str(cfg.assets_dir),
        "steps": cfg.steps,
        "complete_regions": [r for r in cfg.region_list if r not in missing],
        "missing_files": missing,
    }
