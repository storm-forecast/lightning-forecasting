"""Builds the model input dataset and the precipitation dataset.

The parameter files (one per lead time) are concatenated into a single
continuous 6-hourly time series. Files are read one at a time, loading only
the required variables into memory, so there is no dask dependency and the
container's memory stays predictable.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from app.core.regions import MODEL_VARIABLES, RENAME_TO_MODEL

log = logging.getLogger(__name__)

PRECIP_VAR = "tp"


def _open_single(path: Path, wanted: set[str]) -> xr.Dataset:

    ds = xr.open_dataset(path, engine="netcdf4")

    rename = {k: v for k, v in RENAME_TO_MODEL.items() if k in ds.variables}
    ds = ds.rename(rename)

    keep = [v for v in ds.data_vars if v in wanted]
    missing = wanted - set(keep)
    if missing:
        log.warning("%s is missing variables %s", path.name, sorted(missing))

    out = ds[keep].load()
    out = out.drop_vars(["run_time", "valid_time"], errors="ignore")
    ds.close()
    return out


def build_input_dataset(
    param_files: list[Path],
) -> tuple[xr.Dataset, xr.Dataset]:
    """Concatenate the parameter files.

    Returns (ds_synoptic, ds_precip):
      - ds_synoptic: the 15 model input variables
      - ds_precip:   6 h accumulated precipitation (tp), in mm
    """

    if not param_files:
        raise ValueError("No parameter file available.")

    wanted = set(MODEL_VARIABLES) | {PRECIP_VAR}

    pieces = []
    for path in sorted(param_files):
        try:
            pieces.append(_open_single(Path(path), wanted))
        except Exception as error:                              # noqa: BLE001
            log.exception("Failed to open %s: %s", path, error)

    if not pieces:
        raise ValueError("No parameter file could be opened.")

    ds = xr.concat(pieces, dim="time", coords="minimal", compat="override")
    for piece in pieces:
        piece.close()

    ds = ds.sortby("time").sortby("lat").sortby("lon")

    # Drop repeated valid times (if two cycles cover the same hour).
    _, unique_idx = np.unique(ds["time"].values, return_index=True)
    if len(unique_idx) != ds.sizes["time"]:
        log.warning(
            "Dropping %d duplicated valid time(s).",
            ds.sizes["time"] - len(unique_idx),
        )
        ds = ds.isel(time=np.sort(unique_idx))

    # Round the coordinates to exact multiples of 0.25 deg, which avoids
    # floating-point near-duplicates when the regions are merged.
    ds = ds.assign_coords(
        lat=np.round(ds["lat"].astype("float64"), 3),
        lon=np.round(ds["lon"].astype("float64"), 3),
    )

    missing = [v for v in MODEL_VARIABLES if v not in ds.data_vars]
    if missing:
        raise ValueError(f"Input variables missing from the dataset: {missing}")

    ds_synoptic = ds[MODEL_VARIABLES].copy()

    if PRECIP_VAR in ds.data_vars:
        ds_precip = ds[[PRECIP_VAR]].copy()
        ds_precip[PRECIP_VAR].attrs.update(
            units="mm",
            long_name="Precipitation accumulated over 6 h",
            description="Accumulated over the 6 h before the valid time (GFS APCP surface)",
        )
    else:
        log.error("Precipitation (tp) is missing: rainfall products will be skipped.")
        ds_precip = xr.Dataset()

    log.info(
        "Input dataset: %d valid times, grid %dx%d",
        ds_synoptic.sizes["time"], ds_synoptic.sizes["lat"], ds_synoptic.sizes["lon"],
    )

    return ds_synoptic, ds_precip


def times_as_iso(ds: xr.Dataset) -> list[str]:
    return [
        pd.Timestamp(t).tz_localize("UTC").isoformat()
        for t in ds["time"].values
    ]
