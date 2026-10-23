"""Smoothing of the predicted fields, one valid time at a time.

Same recipe as the notebook: a 3x3 median filter (removes isolated points),
cubic resampling onto a finer grid and a Gaussian filter. Points that were
zero on the original grid stay zero (remask), so smoothing never spreads
lightning into places where the model predicted none.
"""

from __future__ import annotations

import logging

import numpy as np
import xarray as xr
from scipy.ndimage import gaussian_filter, median_filter

log = logging.getLogger(__name__)


def smooth_field(
    da: xr.DataArray,
    upscale: int = 4,
    sigma: float = 1.2,
    despeckle: bool = True,
    vmin: float | None = 0.0,
    vmax: float | None = None,
    remask: bool = True,
    max_side: int | None = None,
) -> xr.DataArray:
    """Smooth one 2D field (lat x lon)."""

    da = da.sortby("lat").sortby("lon")
    src = da.values.astype("float64")

    if despeckle:
        src = median_filter(np.nan_to_num(src, nan=0.0), size=3)

    src_da = da.copy(data=src)

    n_lat = da.sizes["lat"] * upscale
    n_lon = da.sizes["lon"] * upscale

    if max_side is not None:
        longest = max(n_lat, n_lon)
        if longest > max_side:
            factor = max_side / longest
            n_lat = max(da.sizes["lat"], int(n_lat * factor))
            n_lon = max(da.sizes["lon"], int(n_lon * factor))

    new_lat = np.linspace(float(da.lat.min()), float(da.lat.max()), n_lat)
    new_lon = np.linspace(float(da.lon.min()), float(da.lon.max()), n_lon)

    fine = src_da.interp(lat=new_lat, lon=new_lon, method="cubic")

    values = gaussian_filter(np.nan_to_num(fine.values, nan=0.0), sigma=sigma)

    if vmin is not None:
        values = np.maximum(values, vmin)
    if vmax is not None:
        values = np.minimum(values, vmax)

    fine = fine.copy(data=values)

    if remask:
        base_zero = src_da.interp(lat=new_lat, lon=new_lon, method="nearest") <= 0
        fine = fine.where(~base_zero, 0.0)

    return fine


def smooth_all_times(da: xr.DataArray, **kwargs) -> xr.DataArray:
    """Apply smooth_field to every valid time."""

    if "time" not in da.dims:
        return smooth_field(da, **kwargs)

    fields = []
    total = da.sizes["time"]

    for i in range(total):
        fields.append(smooth_field(da.isel(time=i), **kwargs))
        log.debug("%s: valid time %d/%d", da.name, i + 1, total)

    return xr.concat(fields, dim="time").assign_coords(time=da["time"].values)


def smooth_dataset(
    ds: xr.Dataset,
    config: dict[str, dict],
    max_side: int | None = None,
) -> xr.Dataset:
    """Smooth the variables described by config -> {variable: kwargs}."""

    output = {}

    for var, kwargs in config.items():

        if var not in ds.data_vars:
            continue

        log.info("Smoothing %s...", var)

        field = smooth_all_times(ds[var], max_side=max_side, **kwargs)
        field.attrs = dict(ds[var].attrs)
        field.attrs["smoothing"] = (
            f"3x3 median + {kwargs.get('upscale', 4)}x upscale (cubic) "
            f"+ Gaussian sigma={kwargs.get('sigma', 1.2)}"
        )
        output[var] = field

        log.info(
            "%s: original max %.4f -> smoothed max %.4f",
            var, float(ds[var].max()), float(field.max()),
        )

    out = xr.Dataset(output)
    out.attrs = dict(ds.attrs)
    return out
