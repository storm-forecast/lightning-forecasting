"""Daily run orchestration.

Steps:
    1. make sure the models are available (Google Drive -> disk)
    2. download the 00z GFS (two days ago, yesterday, today)
    3. compute the atmospheric parameters
    4. build the input dataset and the precipitation dataset
    5. predict lightning per region (classification + regression)
    6. merge the regions by maximum and smooth
    7. generate the precipitation files
    8. write the manifest and move the 'latest' pointer
"""

from __future__ import annotations

import logging
import shutil
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr

from app.config import Settings, settings as default_settings
from app.core import drive, precipitation, state
from app.core.dataset import build_input_dataset
from app.core.gfs import GFSDownloader
from app.core.parameters import compute_parameters_for_files
from app.core.predict import (
    forecast_to_dataarray,
    predict_classification,
    predict_regressor,
)
from app.core.regions import (
    MODEL_VARIABLES,
    VAR_DETER,
    VAR_PROB,
    get_region_bounds,
    region_meta,
)
from app.core.smoothing import smooth_dataset

log = logging.getLogger(__name__)

_run_lock = threading.Lock()

MERGED_NAME = "gfs_prediction_ALL_regions_max.nc"
SMOOTHED_NAME = "gfs_prediction_ALL_regions_suavisado.nc"
PRECIP_NAME = "gfs_precipitation.nc"
PRECIP_SMOOTHED_NAME = "gfs_precipitation_suavisado.nc"


class PipelineBusy(RuntimeError):
    """A run is already in progress."""


def _iso(value) -> str:
    return pd.Timestamp(value).tz_localize("UTC").isoformat()


def _netcdf_encoding(ds: xr.Dataset, dtype: str = "float32") -> dict:
    return {
        var: {"zlib": True, "complevel": 4, "dtype": dtype}
        for var in ds.data_vars
    }


def _save(ds: xr.Dataset, path: Path, dtype: str = "float32") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.nc")
    ds.to_netcdf(tmp, encoding=_netcdf_encoding(ds, dtype))
    tmp.replace(path)
    return path


# ============================================================
# Single-region prediction
# ============================================================


def predict_region(
    ds_synoptic: xr.Dataset,
    region: str,
    cfg: Settings,
) -> xr.Dataset | None:
    """Generate prob-flash-gfs and deter-flash-gfs for one region."""

    meta = region_meta(region)

    minx, miny, maxx, maxy = get_region_bounds(region, cfg.shapefiles_root)
    log.info(
        "%s (%s): lon %.2f to %.2f, lat %.2f to %.2f",
        region, meta.label, minx, maxx, miny, maxy,
    )

    ds = (
        ds_synoptic.sortby("lat").sortby("lon")
        .sel(lon=slice(minx, maxx), lat=slice(miny, maxy))
        .sortby("time")
    )

    if ds.sizes["lat"] == 0 or ds.sizes["lon"] == 0:
        log.error("%s: empty subset, region falls outside the downloaded domain.", region)
        return None

    class_paths = drive.classification_paths(cfg, region)
    reg_paths = drive.regressor_paths(cfg, region)

    predictions_class, _ = predict_classification(
        ds, MODEL_VARIABLES,
        model_path=class_paths["model"],
        mean_path=class_paths["mean"],
        std_path=class_paths["std"],
        window_size=cfg.classifier_window,
        batch_size=cfg.predict_batch_size,
    )

    ds_class = forecast_to_dataarray(predictions_class, ds, VAR_PROB).to_dataset()

    threshold = cfg.prob_threshold
    ds_class[VAR_PROB] = ds_class[VAR_PROB].where(ds_class[VAR_PROB] > threshold, 0.0)

    predictions_reg, _ = predict_regressor(
        ds, MODEL_VARIABLES,
        model_path=reg_paths["model"],
        mean_label_path=reg_paths["mean_label"],
        std_label_path=reg_paths["std_label"],
        mean_data_path=reg_paths["mean_data"],
        std_data_path=reg_paths["std_data"],
        window_size=cfg.regressor_window,
        batch_size=cfg.predict_batch_size,
    )

    ds_reg = forecast_to_dataarray(predictions_reg, ds, VAR_DETER).to_dataset()

    # The classifier and the regressor may use different windows, so keep
    # only the valid times present in both.
    common = np.intersect1d(ds_class["time"].values, ds_reg["time"].values)

    if len(common) == 0:
        log.error("%s: no valid time shared by the classifier and the regressor.", region)
        return None

    ds_final = xr.merge(
        [ds_class.sel(time=common), ds_reg.sel(time=common)], join="exact",
    )

    # Drop valid times that are entirely null.
    mask = ~ds_final[VAR_DETER].isnull().all(dim=["lat", "lon"])
    ds_final = ds_final.sel(time=mask)

    # With no probability above the threshold there is no predicted lightning.
    ds_final[VAR_DETER] = ds_final[VAR_DETER].where(ds_final[VAR_PROB] != 0, 0)

    ds_final[VAR_DETER] = ds_final[VAR_DETER].astype(np.int32)
    ds_final[VAR_PROB] = ds_final[VAR_PROB].astype(np.float32)

    ds_final.attrs.update(
        region=region,
        label=meta.label,
        probability_threshold=threshold,
    )

    log.info(
        "%s: %d valid times predicted (%s to %s)",
        region, ds_final.sizes["time"],
        _iso(ds_final["time"].values[0]), _iso(ds_final["time"].values[-1]),
    )

    return ds_final


# ============================================================
# Region merge
# ============================================================


def merge_regions(datasets: list[xr.Dataset]) -> xr.Dataset:
    """Combine the regions, taking the maximum where they overlap."""

    prepared = [
        ds[[VAR_PROB, VAR_DETER]].assign_coords(
            lat=np.round(ds["lat"].astype("float64"), 3),
            lon=np.round(ds["lon"].astype("float64"), 3),
        )
        for ds in datasets
    ]

    combined = xr.concat(prepared, dim="region", join="outer")
    merged = combined.max(dim="region", skipna=True)

    # No coverage means no lightning; probability is zeroed where the
    # deterministic field is 0.
    deter = merged[VAR_DETER].fillna(0.0)
    prob = merged[VAR_PROB].where(deter != 0, 0.0).fillna(0.0)

    merged[VAR_DETER] = deter.astype("float32")
    merged[VAR_PROB] = prob.astype("float32")

    merged[VAR_PROB].attrs.update(
        long_name="Probability of lightning occurrence in 6 h", units="1",
    )
    merged[VAR_DETER].attrs.update(
        long_name="Predicted flash count in 6 h", units="flashes/6h",
    )

    return merged.sortby(["time", "lat", "lon"])


# ============================================================
# Full run
# ============================================================


def run_pipeline(
    cfg: Settings | None = None,
    reference_time: datetime | None = None,
    force: bool = False,
) -> dict[str, Any]:
    """Run the whole daily pipeline. Returns the manifest."""

    cfg = cfg or default_settings

    if not _run_lock.acquire(blocking=False):
        raise PipelineBusy("A run is already in progress.")

    started = time.monotonic()
    cfg.ensure_dirs()

    run_state = state.RunState(cfg.state_file)
    run_id: str | None = None

    try:
        reference = reference_time or datetime.now(timezone.utc)

        downloader = GFSDownloader(
            output_base=cfg.work_dir / "gfs",
            reference_time=reference,
            horizon_hours=cfg.horizon_hours,
            lon_range=(cfg.lon_min, cfg.lon_max),
            lat_range=(cfg.lat_min, cfg.lat_max),
            max_workers=cfg.gfs_max_workers,
            cleanup_grib=cfg.cleanup_grib,
        )

        run_date = downloader.run_date
        run_id = state.run_id_for(run_date)

        existing = state.get_manifest(cfg.products_dir, run_id)
        if existing and not force:
            log.info("Run %s already processed; pass force=true to redo it.", run_id)
            return existing

        run_state.start(run_id)

        # ---------- 1. models ----------
        run_state.step("Checking the trained models")
        drive.ensure_assets(cfg, force=cfg.refresh_assets_each_run)

        # ---------- 2. GFS ----------
        run_state.step("Downloading the 00z GFS")
        gfs_files = downloader.run()

        if not gfs_files:
            raise RuntimeError(
                "No GFS file available. The 00z cycle has most likely not been "
                "published in the NOAA bucket yet."
            )

        # ---------- 3. parameters ----------
        run_state.step(f"Computing the atmospheric parameters ({len(gfs_files)} lead times)")
        param_dir = cfg.work_dir / "parameters"
        param_files = compute_parameters_for_files(gfs_files, param_dir)

        # ---------- 4. input dataset ----------
        run_state.step("Building the input dataset")
        ds_synoptic, ds_precip = build_input_dataset(param_files)

        input_times = [_iso(t) for t in ds_synoptic["time"].values]

        # ---------- 5. per-region prediction ----------
        lightning_dir = state.run_dir(cfg.products_dir, run_id) / state.LIGHTNING_SUBDIR
        lightning_dir.mkdir(parents=True, exist_ok=True)

        regions_ok: list[str] = []
        regions_failed: dict[str, str] = {}
        region_datasets: list[xr.Dataset] = []
        region_files: dict[str, str] = {}

        missing = drive.missing_assets(cfg, cfg.region_list)

        for region in cfg.region_list:

            if region in missing:
                regions_failed[region] = "models missing on Drive"
                log.warning("%s: models missing, region skipped.", region)
                continue

            run_state.step(f"Predicting lightning: {region}")

            try:
                ds_region = predict_region(ds_synoptic, region, cfg)
            except Exception as error:                          # noqa: BLE001
                log.exception("%s: prediction failed", region)
                regions_failed[region] = str(error)
                continue

            if ds_region is None:
                regions_failed[region] = "no valid time predicted"
                continue

            path = lightning_dir / f"gfs_prediction_{region}.nc"
            _save(ds_region, path)

            region_datasets.append(ds_region)
            region_files[region] = f"{state.LIGHTNING_SUBDIR}/{path.name}"
            regions_ok.append(region)

        if not region_datasets:
            raise RuntimeError(
                "No region produced a prediction. Details: "
                + "; ".join(f"{k}: {v}" for k, v in regions_failed.items())
            )

        # ---------- 6. merge and smoothing ----------
        run_state.step(f"Merging {len(region_datasets)} regions")
        merged = merge_regions(region_datasets)
        merged.attrs.update(
            source="GFS",
            run=f"{run_date:%Y-%m-%d} 00 UTC",
            regions=", ".join(regions_ok),
            description="Per-region lightning forecast, merged by maximum",
        )
        merged_path = _save(merged, lightning_dir / MERGED_NAME)

        for ds_region in region_datasets:
            ds_region.close()

        run_state.step("Smoothing the lightning fields")
        smoothed = smooth_dataset(
            merged,
            {
                VAR_DETER: dict(
                    upscale=cfg.smooth_upscale, sigma=cfg.smooth_sigma,
                    despeckle=True, vmin=0.0, vmax=None, remask=True,
                ),
                VAR_PROB: dict(
                    upscale=cfg.smooth_upscale, sigma=cfg.smooth_sigma,
                    despeckle=True, vmin=0.0, vmax=1.0, remask=True,
                ),
            },
            max_side=cfg.smooth_max_side,
        )
        smoothed.attrs.update(merged.attrs)
        smoothed.attrs["description"] = (
            "Lightning fields smoothed one valid time at a time "
            f"(derived from {MERGED_NAME})"
        )
        smoothed_path = _save(smoothed, lightning_dir / SMOOTHED_NAME)

        lightning_times = [_iso(t) for t in merged["time"].values]
        lightning_bounds = [
            float(merged["lon"].min()), float(merged["lat"].min()),
            float(merged["lon"].max()), float(merged["lat"].max()),
        ]
        lightning_stats = {
            VAR_PROB: {
                "min": float(merged[VAR_PROB].min()),
                "max": float(merged[VAR_PROB].max()),
            },
            VAR_DETER: {
                "min": float(merged[VAR_DETER].min()),
                "max": float(merged[VAR_DETER].max()),
            },
        }

        merged.close()
        smoothed.close()

        # ---------- 7. precipitation ----------
        precip_entry: dict[str, Any] | None = None

        if ds_precip.data_vars:

            run_state.step("Generating the precipitation files")
            precip_dir = state.run_dir(cfg.products_dir, run_id) / state.PRECIP_SUBDIR

            ds_precipitation = precipitation.build_precipitation_dataset(
                ds_precip, run_date, accum_24h=cfg.precip_accum_24h,
            )
            precip_path = precipitation.save_precipitation(
                ds_precipitation, precip_dir / PRECIP_NAME,
            )

            files = {"netcdf": f"{state.PRECIP_SUBDIR}/{precip_path.name}"}

            if cfg.smooth_precipitation:
                ds_precip_smooth = smooth_dataset(
                    ds_precipitation[[precipitation.VAR_6H]],
                    {
                        precipitation.VAR_6H: dict(
                            upscale=cfg.smooth_upscale, sigma=cfg.smooth_sigma,
                            despeckle=True, vmin=0.0, vmax=None, remask=True,
                        ),
                    },
                    max_side=cfg.smooth_max_side,
                )
                ds_precip_smooth.attrs.update(ds_precipitation.attrs)
                precip_smooth_path = precipitation.save_precipitation(
                    ds_precip_smooth, precip_dir / PRECIP_SMOOTHED_NAME,
                )
                files["netcdf_suavizado"] = (
                    f"{state.PRECIP_SUBDIR}/{precip_smooth_path.name}"
                )
                ds_precip_smooth.close()

            precip_entry = {
                "variables": list(ds_precipitation.data_vars),
                "units": "mm",
                "times": [_iso(t) for t in ds_precipitation["time"].values],
                "bounds": [
                    float(ds_precipitation["lon"].min()),
                    float(ds_precipitation["lat"].min()),
                    float(ds_precipitation["lon"].max()),
                    float(ds_precipitation["lat"].max()),
                ],
                "grid": {
                    "lat": int(ds_precipitation.sizes["lat"]),
                    "lon": int(ds_precipitation.sizes["lon"]),
                    "resolution_deg": 0.25,
                },
                "stats": {
                    precipitation.VAR_6H: {
                        "max": float(ds_precipitation[precipitation.VAR_6H].max()),
                    },
                    precipitation.VAR_TOTAL: {
                        "max": float(ds_precipitation[precipitation.VAR_TOTAL].max()),
                    },
                },
                "files": files,
            }

            ds_precipitation.close()

        ds_synoptic.close()
        ds_precip.close()

        # ---------- 8. manifest ----------
        manifest: dict[str, Any] = {
            "run_id": run_id,
            "run_time": f"{run_date:%Y-%m-%dT00:00:00}Z",
            "generated_at": state.utcnow_iso(),
            "duration_seconds": round(time.monotonic() - started, 1),
            "source": "GFS 0.25 deg (NOAA), 00 UTC cycle",
            "horizon_hours": cfg.horizon_hours,
            "input_times": input_times,
            "regions": {
                "processed": regions_ok,
                "failed": regions_failed,
                "labels": {r: region_meta(r).label for r in regions_ok},
            },
            "products": {
                "lightning": {
                    "variables": [VAR_PROB, VAR_DETER],
                    "threshold": cfg.prob_threshold,
                    "times": lightning_times,
                    "bounds": lightning_bounds,
                    "stats": lightning_stats,
                    "files": {
                        "netcdf": f"{state.LIGHTNING_SUBDIR}/{merged_path.name}",
                        "netcdf_suavizado": (
                            f"{state.LIGHTNING_SUBDIR}/{smoothed_path.name}"
                        ),
                        "por_regiao": region_files,
                    },
                },
            },
        }

        if precip_entry is not None:
            manifest["products"]["precipitation"] = precip_entry

        state.save_manifest(cfg.products_dir, run_id, manifest)
        state.set_latest(cfg.products_dir, run_id)
        state.prune_runs(cfg.products_dir, cfg.keep_runs)

        if cfg.cleanup_workdir:
            shutil.rmtree(cfg.work_dir, ignore_errors=True)

        run_state.finish(run_id)
        log.info(
            "Run %s finished in %.1f s (%d regions)",
            run_id, manifest["duration_seconds"], len(regions_ok),
        )

        return manifest

    except Exception as error:                                  # noqa: BLE001
        log.exception("Run failed")
        state.RunState(cfg.state_file).fail(run_id, f"{type(error).__name__}: {error}")
        raise

    finally:
        _run_lock.release()


def pipeline_is_running() -> bool:
    return _run_lock.locked()


def next_run_datetime(cfg: Settings, now: datetime | None = None) -> datetime:
    """Next scheduled time, in UTC."""
    now = now or datetime.now(timezone.utc)
    hour, minute = (int(p) for p in cfg.run_at_utc.split(":"))
    candidate = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
    if candidate <= now:
        candidate += timedelta(days=1)
    return candidate
