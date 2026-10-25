"""Daily run orchestration.

Steps:
    1. make sure the models are available (Google Drive -> disk)
    2. download the 00z cycle of the chosen source (two days ago, yesterday, today)
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
from app.core.ecmwf import ECMWFDownloader
from app.core.gfs import GFSDownloader
from app.core.parameters import compute_parameters_for_files
from app.core.predict import (
    forecast_to_dataarray,
    predict_classification,
    predict_regressor,
)
from app.core.regions import (
    MODEL_VARIABLES,
    SOURCE_LABEL,
    get_region_bounds,
    region_meta,
    source_vars,
)
from app.core.smoothing import smooth_dataset

log = logging.getLogger(__name__)

_run_lock = threading.Lock()

# The downloader for each source. Both have the same constructor and the same `run()`, which is
# what lets everything below this line be written once.
DOWNLOADERS = {"gfs": GFSDownloader, "ecmwf": ECMWFDownloader}

# File name prefix per source. "gfs" is the prefix the release has always carried, so it stays;
# the IFS products sit beside it rather than replacing it, and a run can hold both.
PREFIX = {"gfs": "gfs", "ecmwf": "ifs"}


def product_names(source: str) -> dict[str, str]:
    prefix = PREFIX[source]
    return {
        "merged": f"{prefix}_prediction_ALL_regions_max.nc",
        "smoothed": f"{prefix}_prediction_ALL_regions_suavisado.nc",
        "precip": f"{prefix}_precipitation.nc",
        "precip_smoothed": f"{prefix}_precipitation_suavisado.nc",
        "precip_global": f"{prefix}_precipitation_global.nc",
    }


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
    source: str = "gfs",
) -> xr.Dataset | None:
    """Generate the probability and deterministic fields for one region."""

    var_prob, var_deter = source_vars(source)
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

    ds_class = forecast_to_dataarray(predictions_class, ds, var_prob).to_dataset()

    threshold = cfg.prob_threshold
    ds_class[var_prob] = ds_class[var_prob].where(ds_class[var_prob] > threshold, 0.0)

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

    ds_reg = forecast_to_dataarray(predictions_reg, ds, var_deter).to_dataset()

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
    mask = ~ds_final[var_deter].isnull().all(dim=["lat", "lon"])
    ds_final = ds_final.sel(time=mask)

    # With no probability above the threshold there is no predicted lightning.
    ds_final[var_deter] = ds_final[var_deter].where(ds_final[var_prob] != 0, 0)

    ds_final[var_deter] = ds_final[var_deter].astype(np.int32)
    ds_final[var_prob] = ds_final[var_prob].astype(np.float32)

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


def merge_regions(datasets: list[xr.Dataset], source: str = "gfs") -> xr.Dataset:
    """Combine the regions, taking the maximum where they overlap."""

    var_prob, var_deter = source_vars(source)

    prepared = [
        ds[[var_prob, var_deter]].assign_coords(
            lat=np.round(ds["lat"].astype("float64"), 3),
            lon=np.round(ds["lon"].astype("float64"), 3),
        )
        for ds in datasets
    ]

    combined = xr.concat(prepared, dim="region", join="outer")
    merged = combined.max(dim="region", skipna=True)

    # No coverage means no lightning; probability is zeroed where the
    # deterministic field is 0.
    deter = merged[var_deter].fillna(0.0)
    prob = merged[var_prob].where(deter != 0, 0.0).fillna(0.0)

    merged[var_deter] = deter.astype("float32")
    merged[var_prob] = prob.astype("float32")

    merged[var_prob].attrs.update(
        long_name="Probability of lightning occurrence in 6 h", units="1",
    )
    merged[var_deter].attrs.update(
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
    source: str = "gfs",
) -> dict[str, Any]:
    """Run the whole daily pipeline for one source. Returns the manifest.

    Called once per source. A second call for the same run_id adds its products to the manifest
    already on disk rather than replacing it, so one run directory holds both GFS and IFS.
    """

    cfg = cfg or default_settings
    var_prob, var_deter = source_vars(source)
    names = product_names(source)

    if not _run_lock.acquire(blocking=False):
        raise PipelineBusy("A run is already in progress.")

    started = time.monotonic()
    cfg.ensure_dirs()

    run_state = state.RunState(cfg.state_file)
    run_id: str | None = None

    try:
        reference = reference_time or datetime.now(timezone.utc)

        downloader = DOWNLOADERS[source](
            output_base=cfg.work_dir / source,
            reference_time=reference,
            horizon_hours=cfg.horizon_hours,
            lon_range=(cfg.lon_min, cfg.lon_max),
            lat_range=(cfg.lat_min, cfg.lat_max),
            max_workers=cfg.gfs_max_workers,
            cleanup_grib=cfg.cleanup_grib,
        )

        run_date = downloader.run_date
        run_id = state.run_id_for(run_date)

        existing = state.get_manifest(cfg.products_dir, run_id) or {}
        if existing.get("sources", {}).get(source) and not force:
            log.info(
                "Run %s already has %s; pass force=true to redo it.", run_id, source,
            )
            return existing

        run_state.start(run_id)

        # ---------- 1. models ----------
        run_state.step("Checking the trained models")
        drive.ensure_assets(cfg, force=cfg.refresh_assets_each_run)

        # ---------- 2. input data ----------
        run_state.step(f"Downloading the 00z {source.upper()}")
        gfs_files = downloader.run()

        if not gfs_files:
            raise RuntimeError(
                f"No {source.upper()} file available. The 00z cycle has most likely not "
                "been published in the bucket yet."
            )

        # ---------- 3. parameters ----------
        run_state.step(f"Computing the atmospheric parameters ({len(gfs_files)} lead times)")
        param_dir = cfg.work_dir / f"parameters-{source}"
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
                ds_region = predict_region(ds_synoptic, region, cfg, source=source)
            except Exception as error:                          # noqa: BLE001
                log.exception("%s: prediction failed", region)
                regions_failed[region] = str(error)
                continue

            if ds_region is None:
                regions_failed[region] = "no valid time predicted"
                continue

            path = lightning_dir / f"{PREFIX[source]}_prediction_{region}.nc"
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
        merged = merge_regions(region_datasets, source=source)
        merged.attrs.update(
            source=SOURCE_LABEL[source],
            run=f"{run_date:%Y-%m-%d} 00 UTC",
            regions=", ".join(regions_ok),
            description="Per-region lightning forecast, merged by maximum",
        )
        merged_path = _save(merged, lightning_dir / names["merged"])

        for ds_region in region_datasets:
            ds_region.close()

        run_state.step("Smoothing the lightning fields")
        smoothed = smooth_dataset(
            merged,
            {
                var_deter: dict(
                    upscale=cfg.smooth_upscale, sigma=cfg.smooth_sigma,
                    despeckle=True, vmin=0.0, vmax=None, remask=True,
                ),
                var_prob: dict(
                    upscale=cfg.smooth_upscale, sigma=cfg.smooth_sigma,
                    despeckle=True, vmin=0.0, vmax=1.0, remask=True,
                ),
            },
            max_side=cfg.smooth_max_side,
        )
        smoothed.attrs.update(merged.attrs)
        smoothed.attrs["description"] = (
            "Lightning fields smoothed one valid time at a time "
            f"(derived from {names['merged']})"
        )
        smoothed_path = _save(smoothed, lightning_dir / names["smoothed"])

        lightning_times = [_iso(t) for t in merged["time"].values]
        lightning_bounds = [
            float(merged["lon"].min()), float(merged["lat"].min()),
            float(merged["lon"].max()), float(merged["lat"].max()),
        ]
        lightning_stats = {
            var_prob: {
                "min": float(merged[var_prob].min()),
                "max": float(merged[var_prob].max()),
            },
            var_deter: {
                "min": float(merged[var_deter].min()),
                "max": float(merged[var_deter].max()),
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
                ds_precipitation, precip_dir / names["precip"],
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
                    ds_precip_smooth, precip_dir / names["precip_smoothed"],
                )
                files["netcdf_suavizado"] = (
                    f"{state.PRECIP_SUBDIR}/{precip_smooth_path.name}"
                )
                ds_precip_smooth.close()

            # ---------- 7b. precipitation, worldwide ----------
            #
            # Same product, no region. It exists because precipitation comes straight out of the
            # forecast model and the model is global: the South American crop is a choice the
            # lightning models force, not one the rain needs.
            #
            # A separate download rather than a wider one for everything: the models are trained per
            # region and cannot be applied elsewhere, while the fifteen parameters that feed them would
            # be computed over eighteen times the points to then be discarded outside those regions.
            # And it reads the downloaded files straight, skipping compute_parameters_for_files, which
            # needs the full field set and would fail on a download carrying only `tp`.
            if cfg.precip_global:
                run_state.step("Generating the worldwide precipitation")
                try:
                    mundo = DOWNLOADERS[source](
                        output_base=cfg.work_dir / f"{source}_global",
                        reference_time=reference,
                        horizon_hours=cfg.horizon_hours,
                        lon_range=(cfg.precip_global_lon_min, cfg.precip_global_lon_max),
                        lat_range=(cfg.precip_global_lat_min, cfg.precip_global_lat_max),
                        max_workers=cfg.gfs_max_workers,
                        cleanup_grib=True,
                        only=["tp"],
                    )
                    arquivos = mundo.run()
                    if not arquivos:
                        raise RuntimeError("the worldwide download produced no file")

                    # The downloader writes the GRIB names, `latitude` and `longitude`; everything
                    # downstream of compute_parameters_for_files sees `lat` and `lon` because that step
                    # renames them. This path skips that step, so it renames them here -- without it
                    # build_precipitation_dataset sorts by a coordinate that does not exist.
                    partes = []
                    for f in sorted(arquivos, key=lambda f: f.valid_time):
                        parte = xr.open_dataset(f.path)[["tp"]]
                        renomear = {k: v for k, v in (("latitude", "lat"), ("longitude", "lon"))
                                    if k in parte.coords or k in parte.dims}
                        partes.append(parte.rename(renomear) if renomear else parte)
                    ds_mundo = xr.concat(partes, dim="time", coords="minimal", compat="override")
                    for parte in partes:
                        parte.close()

                    ds_global = precipitation.build_precipitation_dataset(
                        ds_mundo, run_date, accum_24h=cfg.precip_accum_24h,
                    )
                    global_path = precipitation.save_precipitation(
                        ds_global, precip_dir / names["precip_global"],
                    )
                    files["netcdf_global"] = f"{state.PRECIP_SUBDIR}/{global_path.name}"
                    log.info("worldwide precipitation: %s (%.1f MB, %s)", global_path.name,
                             global_path.stat().st_size / 1e6, dict(ds_global.sizes))
                    ds_global.close()
                    ds_mundo.close()
                except Exception as error:                              # noqa: BLE001
                    # The regional products are the ones that matter; a failure out here must not cost
                    # them the run they already finished.
                    log.warning("worldwide precipitation failed, skipped: %s", error)

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
        products: dict[str, Any] = {
            "lightning": {
                "variables": [var_prob, var_deter],
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
        }

        if precip_entry is not None:
            products["precipitation"] = precip_entry

        entry: dict[str, Any] = {
            "source": SOURCE_LABEL[source],
            "generated_at": state.utcnow_iso(),
            "duration_seconds": round(time.monotonic() - started, 1),
            "input_times": input_times,
            "regions": {
                "processed": regions_ok,
                "failed": regions_failed,
                "labels": {r: region_meta(r).label for r in regions_ok},
            },
            "products": products,
        }

        # Read again rather than reuse `existing`: the other source may have finished and saved
        # while this one was running, and its products must not be dropped.
        manifest: dict[str, Any] = dict(state.get_manifest(cfg.products_dir, run_id) or {})
        manifest["run_id"] = run_id
        manifest["run_time"] = f"{run_date:%Y-%m-%dT00:00:00}Z"
        manifest["horizon_hours"] = cfg.horizon_hours
        manifest.setdefault("sources", {})[source] = entry

        # The top level keeps meaning GFS, because that is what /api/v1/status, /lightning and the
        # map already read. The per-source detail lives under "sources"; a run with no GFS at all
        # fills the top level from whichever source did run, so those endpoints still answer.
        if source == "gfs" or "products" not in manifest:
            manifest.update(
                {k: v for k, v in entry.items() if k != "products"},
                products=products,
            )

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
