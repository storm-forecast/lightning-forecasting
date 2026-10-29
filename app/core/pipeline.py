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

# File name prefix per source. "gfs" is the prefix the release has always carried, so it stays; the
# ECMWF products sit beside it rather than replacing it, and a run can hold both. The ECMWF files are
# called "ecens" because that is the name the page shows the reader -- "ifs" is the model's name inside
# the centre, and a file named after what nobody calls it is a file somebody has to translate.
PREFIX = {"gfs": "gfs", "ecmwf": "ecens"}


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
    """Compressed, loss-bounded representation for forecast products.

    NetCDF's ``scale_factor`` writes a small integer but xarray restores the
    physical value on read.  A probability therefore keeps two decimal places
    *in percent* (43.27% becomes 4327) instead of spending four bytes on a
    float, and the deterministic flash field keeps tenths after smoothing.
    The latter is deliberately not rounded to a whole flash: smoothing turns a
    count into a continuous spatial field.
    """
    encoding = {var: {"zlib": True, "complevel": 4, "dtype": dtype}
                for var in ds.data_vars}

    for var in ds.data_vars:
        if var.startswith("prob-flash-"):
            _pack_uint16(encoding[var], ds[var], 0.0001, var)
        elif var.startswith("deter-flash-"):
            _pack_uint16(encoding[var], ds[var], 0.1, var)

    # The grid is precise to 0.001° before smoothing, so float32 is still over
    # a hundred times finer than the source grid.  Coordinates otherwise stay
    # float64 in every released product.
    for coordinate in ("lat", "lon"):
        if coordinate in ds.coords:
            encoding[coordinate] = {"dtype": "float32"}
    return encoding


def _pack_uint16(encoding: dict, values: xr.DataArray, scale: float, name: str) -> None:
    """Use uint16 packing when its physical range contains every value.

    Falling back is preferable to clipping an unusually intense forecast.
    ``65535`` remains reserved as the NetCDF missing-value marker.
    """
    maximum = float(values.max(skipna=True))
    if np.isfinite(maximum) and maximum <= 65534 * scale:
        encoding.update(dtype="uint16", scale_factor=scale,
                        _FillValue=np.uint16(65535))
    else:
        log.warning("%s reaches %s, outside uint16 packing at scale %s; keeping float32",
                    name, maximum, scale)


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
# One time axis for every source
#
# The two sources must publish the same valid times, or a page that steps through both frame by
# frame breaks. They can diverge for two reasons: a step missing from one bucket (the ECMWF open
# data keeps only about four days and lands later than the GFS), and the all-null drop in
# predict_region, which depends on the data rather than on the configuration.
#
# Precipitation has to be aligned BEFORE it is built: precip_24h is a rolling sum of four and
# precip_total covers the window, so trimming time afterwards would leave both lying about what
# they accumulate. Lightning can be aligned afterwards, because merging and smoothing work one
# valid time at a time. Hence two mechanisms: restrict_times on the way in, and
# align_lightning_times as the net underneath.
# ============================================================


def prepare_inputs(
    cfg: Settings,
    reference_time: datetime | None,
    source: str,
) -> pd.DatetimeIndex:
    """Download one source and compute its parameters, returning the valid times it can offer.

    Cheap to call twice: the downloader skips lead times whose NetCDF is already on disk and
    compute_parameters_for_files skips parameters already computed, so the run that follows reuses
    all of this instead of fetching it again.
    """

    downloader = DOWNLOADERS[source](
        output_base=cfg.work_dir / source,
        reference_time=reference_time,
        horizon_hours=cfg.horizon_hours,
        lon_range=(cfg.lon_min, cfg.lon_max),
        lat_range=(cfg.lat_min, cfg.lat_max),
        max_workers=cfg.gfs_max_workers,
        cleanup_grib=cfg.cleanup_grib,
    )

    files = downloader.run()
    if not files:
        raise RuntimeError(
            f"No {source.upper()} file available. The 00z cycle has most likely not been "
            "published in the bucket yet."
        )

    compute_parameters_for_files(files, cfg.work_dir / f"parameters-{source}")

    # The valid time of each file is the time axis build_input_dataset will produce, so there is
    # nothing to open to find it out.
    #
    # A DatetimeIndex rather than datetime64 in a set, here and everywhere these times are
    # compared: datetime64 hashes by unit, so a [s] value and the identical [ns] value compare
    # equal and hash differently, and `in set` silently answers False. That would turn every
    # alignment below into a no-op that logs nothing.
    return pd.DatetimeIndex(
        sorted({f.valid_time.replace(tzinfo=None) for f in files})
    )


def common_times(
    per_source: dict[str, pd.DatetimeIndex],
) -> pd.DatetimeIndex | None:
    """The valid times every source can offer, or None when restricting would not help."""

    if len(per_source) < 2:
        return None

    axes = list(per_source.values())
    common = axes[0]
    for other in axes[1:]:
        common = common.intersection(other)

    if len(common) == 0:
        log.warning(
            "The sources share no valid time (%s); each will run on its own series.",
            {k: len(v) for k, v in per_source.items()},
        )
        return None

    dropped = {
        source: len(times) - len(common)
        for source, times in per_source.items()
        if len(times) != len(common)
    }
    if dropped:
        log.warning("Aligning the sources drops %s lead times to match", dropped)
    else:
        log.info("The sources already agree on %d valid times", len(common))

    return common.sort_values()


def align_lightning_times(cfg: Settings, run_id: str) -> list[str] | None:
    """Trim every source's lightning products to the valid times they share.

    The net under restrict_times, for the all-null drop that happens inside the prediction. A
    no-op in the normal case, where the axes already agree.
    """

    manifest = state.get_manifest(cfg.products_dir, run_id)
    if not manifest:
        return None

    sources = {
        source: entry for source, entry in manifest.get("sources", {}).items()
        if entry.get("products", {}).get("lightning")
    }
    if len(sources) < 2:
        return None

    axes = {
        source: entry["products"]["lightning"]["times"]
        for source, entry in sources.items()
    }
    shared = None
    for times in axes.values():
        index = pd.DatetimeIndex([pd.Timestamp(t).tz_convert(None) for t in times])
        shared = index if shared is None else shared.intersection(index)
    common = [_iso(t) for t in shared.sort_values()]

    if not common:
        log.error(
            "The sources' lightning products share no valid time (%s); left untouched.",
            {k: len(v) for k, v in axes.items()},
        )
        return None

    if any(len(t) != len(common) for t in axes.values()):
        log.warning(
            "Trimming the lightning products to %d shared valid times (from %s)",
            len(common), {k: len(v) for k, v in axes.items()},
        )
    else:
        # Still worth walking through: the manifest should say the axes agree, which is what a
        # consumer checks. Nothing is rewritten in this case but the manifest itself.
        log.info("The lightning products already share %d valid times", len(common))

    run_directory = state.run_dir(cfg.products_dir, run_id)
    # The manifest carries what _iso wrote, which is tz-aware ("+00:00"); the datasets store
    # naive UTC.
    keep = pd.DatetimeIndex([pd.Timestamp(t).tz_convert(None) for t in common])

    for source, entry in sources.items():
        lightning = entry["products"]["lightning"]
        # Recorded on every source, including one that needed no trimming: what a consumer wants
        # to know is that the axes agree, not which file happened to be rewritten.
        lightning["times"] = common
        lightning["aligned_to_sources"] = sorted(sources)

        if len(axes[source]) == len(common):
            continue

        relatives = [lightning["files"]["netcdf"], lightning["files"]["netcdf_suavizado"]]
        relatives += list(lightning["files"].get("por_regiao", {}).values())

        for relative in relatives:
            path = run_directory / relative
            if not path.exists():
                continue
            with xr.open_dataset(path) as handle:
                # A region file only holds the times that region predicted, so intersect per
                # file rather than selecting the run's whole list.
                present = pd.DatetimeIndex(handle["time"].values).intersection(keep)
                trimmed = handle.sel(time=present).load()
            # A dataset read back from disk carries that file's encoding, whose chunk sizes
            # describe the untrimmed time dimension -- writing fewer times with them is rejected
            # by netCDF as an invalid argument.
            for variable in trimmed.variables.values():
                variable.encoding = {}
            _save(trimmed, path)
            trimmed.close()

        var_prob, var_deter = source_vars(source)
        merged_path = run_directory / lightning["files"]["netcdf"]
        with xr.open_dataset(merged_path) as handle:
            lightning["stats"] = {
                var_prob: {
                    "min": float(handle[var_prob].min()),
                    "max": float(handle[var_prob].max()),
                },
                var_deter: {
                    "min": float(handle[var_deter].min()),
                    "max": float(handle[var_deter].max()),
                },
            }

    if manifest.get("products", {}).get("lightning"):
        primary = "gfs" if "gfs" in sources else sorted(sources)[0]
        manifest["products"]["lightning"] = sources[primary]["products"]["lightning"]

    state.save_manifest(cfg.products_dir, run_id, manifest)
    return common


# ============================================================
# Full run
# ============================================================


def run_pipeline(
    cfg: Settings | None = None,
    reference_time: datetime | None = None,
    force: bool = False,
    source: str = "gfs",
    restrict_times: list | None = None,
    cleanup: bool | None = None,
) -> dict[str, Any]:
    """Run the whole daily pipeline for one source. Returns the manifest.

    Called once per source. A second call for the same run_id adds its products to the manifest
    already on disk rather than replacing it, so one run directory holds both GFS and IFS.

    restrict_times limits the run to these valid times, which is how two sources end up publishing
    the same period -- see the section above. cleanup overrides cfg.cleanup_workdir, so an
    orchestrator running several sources can keep the downloads until the last one is done.
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

        if restrict_times is not None:
            # Intersect rather than select: a time this source never had would raise, and the
            # caller's list was built from every source.
            wanted = pd.DatetimeIndex(restrict_times)
            keep = pd.DatetimeIndex(ds_synoptic["time"].values).intersection(wanted)
            if len(keep) == 0:
                raise RuntimeError(
                    f"{source}: none of its valid times is in the aligned set"
                )
            dropped = ds_synoptic.sizes["time"] - len(keep)
            if dropped:
                log.info("%s: %d valid times dropped to align the sources", source, dropped)
            ds_synoptic = ds_synoptic.sel(time=keep)
            ds_precip = ds_precip.sel(
                time=pd.DatetimeIndex(ds_precip["time"].values).intersection(wanted)
            )

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

                    # This pass downloads its own `tp`, so the restriction applied to the regional
                    # datasets never reached it -- and it has to, because the two sources' global
                    # files are published side by side and a page reads them frame by frame. Same
                    # reason as there: the accumulations are built below, so trimming has to
                    # happen first or precip_24h and precip_total would describe a window the file
                    # no longer holds.
                    if restrict_times is not None:
                        wanted = pd.DatetimeIndex(restrict_times)
                        aligned = pd.DatetimeIndex(ds_mundo["time"].values).intersection(wanted)
                        if len(aligned) != len(wanted):
                            # Nothing downstream can repair this one: trimming after the fact
                            # would break the accumulations, and the raw tp is deleted with the
                            # work directory. Publishing a worldwide file on a different axis from
                            # the other source's would break the page the alignment exists for, so
                            # the failure below drops this source's worldwide file instead. A
                            # missing layer a page can handle; a misaligned one it cannot.
                            raise RuntimeError(
                                f"worldwide rain covers {len(aligned)} of the {len(wanted)} "
                                "aligned valid times"
                            )
                        ds_mundo = ds_mundo.sel(time=aligned)

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

        if cfg.cleanup_workdir if cleanup is None else cleanup:
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


def run_precipitation_only(
    cfg: Settings | None = None,
    reference_time: datetime | None = None,
    force: bool = False,
    source: str = "ecmwf",
    cleanup: bool | None = None,
) -> dict[str, Any]:
    """Publish worldwide precipitation from one source without loading lightning models.

    ECMWF remains useful as an independent rain forecast, but its lightning
    fields require the same 3 GB model bundle as GFS.  This path downloads its
    single ``tp`` field and writes the global rain product directly.
    """
    cfg = cfg or default_settings
    if not _run_lock.acquire(blocking=False):
        raise PipelineBusy("A run is already in progress.")

    started = time.monotonic()
    cfg.ensure_dirs()
    run_state = state.RunState(cfg.state_file)
    run_id: str | None = None

    try:
        reference = reference_time or datetime.now(timezone.utc)
        downloader = DOWNLOADERS[source](
            output_base=cfg.work_dir / f"{source}_global",
            reference_time=reference,
            horizon_hours=cfg.horizon_hours,
            lon_range=(cfg.precip_global_lon_min, cfg.precip_global_lon_max),
            lat_range=(cfg.precip_global_lat_min, cfg.precip_global_lat_max),
            max_workers=cfg.gfs_max_workers,
            cleanup_grib=True,
            only=["tp"],
        )
        run_date = downloader.run_date
        run_id = state.run_id_for(run_date)
        existing = state.get_manifest(cfg.products_dir, run_id) or {}
        if existing.get("sources", {}).get(source) and not force:
            log.info("Run %s already has %s precipitation; pass force=true to redo it.", run_id, source)
            return existing

        run_state.start(run_id)
        run_state.step(f"Downloading worldwide {source.upper()} precipitation")
        files = downloader.run()
        if not files:
            raise RuntimeError("the worldwide precipitation download produced no file")

        parts = []
        for item in sorted(files, key=lambda item: item.valid_time):
            part = xr.open_dataset(item.path)[["tp"]]
            rename = {key: value for key, value in (("latitude", "lat"), ("longitude", "lon"))
                      if key in part.coords or key in part.dims}
            parts.append(part.rename(rename) if rename else part)
        world = xr.concat(parts, dim="time", coords="minimal", compat="override")
        for part in parts:
            part.close()

        run_state.step("Generating worldwide precipitation")
        rain = precipitation.build_precipitation_dataset(
            world, run_date, accum_24h=cfg.precip_accum_24h,
        )
        path = precipitation.save_precipitation(
            rain,
            state.run_dir(cfg.products_dir, run_id) / state.PRECIP_SUBDIR / product_names(source)["precip_global"],
        )
        rain.close()
        world.close()

        entry = {
            "source": SOURCE_LABEL[source],
            "generated_at": state.utcnow_iso(),
            "duration_seconds": round(time.monotonic() - started, 1),
            "regions": {"processed": [], "failed": {}, "labels": {}},
            "products": {"precipitation": {
                "variables": [precipitation.VAR_6H, precipitation.VAR_24H, precipitation.VAR_TOTAL],
                "units": "mm",
                "files": {"netcdf_global": f"{state.PRECIP_SUBDIR}/{path.name}"},
            }},
        }
        manifest = dict(state.get_manifest(cfg.products_dir, run_id) or {})
        manifest["run_id"] = run_id
        manifest["run_time"] = f"{run_date:%Y-%m-%dT00:00:00}Z"
        manifest["horizon_hours"] = cfg.horizon_hours
        manifest.setdefault("sources", {})[source] = entry
        state.save_manifest(cfg.products_dir, run_id, manifest)
        state.set_latest(cfg.products_dir, run_id)
        state.prune_runs(cfg.products_dir, cfg.keep_runs)
        run_state.finish(run_id)
        return manifest
    except Exception as error:                                  # noqa: BLE001
        log.exception("Precipitation-only run failed")
        if run_id:
            run_state.fail(run_id, f"{type(error).__name__}: {error}")
        raise
    finally:
        if cfg.cleanup_workdir if cleanup is None else cleanup:
            shutil.rmtree(cfg.work_dir, ignore_errors=True)
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
