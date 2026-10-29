"""GFS precipitation products.

Precipitation does not go through any model: it is the GFS APCP variable
itself (accumulated over the 6 h before each valid time), already subset to
South America during the download. Here it is organised into three fields:

    precip_6h     6 h accumulation at each valid time
    precip_24h    24 h accumulation (sum of the four preceding valid times)
    precip_total  accumulation over today's whole forecast window

Output is NetCDF, sharing the time structure of the lightning products so
the web page can reuse the same valid-time control.
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

log = logging.getLogger(__name__)

VAR_6H = "precip_6h"
VAR_24H = "precip_24h"
VAR_TOTAL = "precip_total"

STEP_HOURS = 6
STEPS_PER_DAY = 4


def build_precipitation_dataset(
    ds_precip: xr.Dataset,
    run_date: datetime,
    accum_24h: bool = True,
) -> xr.Dataset:
    """Build the precipitation dataset from the GFS 'tp' variable."""

    if "tp" not in ds_precip.data_vars:
        raise ValueError("Dataset has no 'tp' variable.")

    ds_precip = ds_precip.sortby("time").sortby("lat").sortby("lon")

    tp = ds_precip["tp"].astype("float32")

    # kg m-2 equals mm; an accumulation is never negative.
    tp = tp.where(tp > 0, 0.0)

    out = xr.Dataset({VAR_6H: tp})
    out[VAR_6H].attrs.update(
        units="mm",
        long_name="Precipitation accumulated over 6 h",
        description="Accumulated over the 6 h before the valid time (GFS APCP surface)",
    )

    # --------------------------------------------------------
    # 24 h accumulation
    # --------------------------------------------------------

    if accum_24h and tp.sizes["time"] >= STEPS_PER_DAY:

        rolling = (
            tp.rolling(time=STEPS_PER_DAY, min_periods=STEPS_PER_DAY)
            .sum()
            .astype("float32")
        )
        out[VAR_24H] = rolling
        out[VAR_24H].attrs.update(
            units="mm",
            long_name="Precipitation accumulated over 24 h",
            description=(
                "Sum of the four 6 h valid times ending at the given valid time; "
                "NaN at the leading valid times, where the window is incomplete"
            ),
        )

    # --------------------------------------------------------
    # Total accumulation over the forecast window (today's cycle)
    # --------------------------------------------------------

    run = pd.Timestamp(run_date).tz_localize(None)
    future = tp.sel(time=tp["time"] > np.datetime64(run))

    if future.sizes["time"] > 0:
        total = future.sum("time").astype("float32")
        start = pd.Timestamp(future["time"].values[0])
        end = pd.Timestamp(future["time"].values[-1])
    else:
        total = tp.sum("time").astype("float32")
        start = pd.Timestamp(tp["time"].values[0])
        end = pd.Timestamp(tp["time"].values[-1])

    out[VAR_TOTAL] = total
    out[VAR_TOTAL].attrs.update(
        units="mm",
        long_name="Precipitation accumulated over the forecast window",
        description=(
            f"Sum from {start:%Y-%m-%d %H} UTC to {end:%Y-%m-%d %H} UTC "
            f"({run:%Y-%m-%d} 00 UTC cycle)"
        ),
        window_start=f"{start:%Y-%m-%dT%H:%M:%S}Z",
        window_end=f"{end:%Y-%m-%dT%H:%M:%S}Z",
    )

    out.attrs.update(
        source="GFS 0.25 deg (NOAA), APCP surface variable",
        run=f"{run:%Y-%m-%d} 00 UTC",
        description="GFS forecast precipitation, with no model post-processing",
    )

    log.info(
        "Precipitation: %d valid times, 6 h max %.1f mm, total max %.1f mm",
        out.sizes["time"], float(tp.max()), float(total.max()),
    )

    return out


def save_precipitation(ds: xr.Dataset, output_file: Path) -> Path:

    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Rain is continuous, so an integer without a scale would erase light
    # precipitation.  One stored unit equals 0.1 mm: enough for the map and
    # point values, while a uint16 still reaches 6553.4 mm in a window.
    encoding = {}
    for var in ds.data_vars:
        field = ds[var]
        maximum = float(field.max(skipna=True))
        if np.isfinite(maximum) and maximum <= 65534 * 0.1:
            encoding[var] = {"zlib": True, "complevel": 4, "dtype": "uint16",
                             "scale_factor": 0.1, "_FillValue": np.uint16(65535)}
        else:
            # An exceptional accumulation must stay exact enough to be useful;
            # never silently wrap it in the compact representation.
            log.warning("%s reaches %s mm, outside uint16 packing; keeping float32", var, maximum)
            encoding[var] = {"zlib": True, "complevel": 4, "dtype": "float32"}
    for coordinate in ("lat", "lon"):
        if coordinate in ds.coords:
            encoding[coordinate] = {"dtype": "float32"}

    tmp = output_file.with_suffix(".tmp.nc")
    ds.to_netcdf(tmp, encoding=encoding)
    tmp.replace(output_file)

    log.info("Precipitation saved: %s (%.1f MB)",
             output_file.name, output_file.stat().st_size / 1e6)

    return output_file
