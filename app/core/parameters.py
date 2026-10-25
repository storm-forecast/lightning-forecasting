"""Atmospheric parameters used as model inputs.

Formulas identical to the training notebook: Total Totals, K index,
divergence at 850 and 250 hPa, low- and mid-level wind shear, thermal
gradient at 850 hPa and dew point.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

log = logging.getLogger(__name__)

# gfs.* or ifs.*: the two downloaders write the same fields under the same names, so the only
# thing that distinguishes their files here is the prefix -- and nothing below needs to know
# which one it is reading.
FILENAME_RE = re.compile(r"(?:gfs|ifs)\.(\d{8})\.00z\.f(\d{3})\.nc$")


class AtmosphericParameters:
    """Reads one source NetCDF (GFS or IFS) and returns the dataset with the indices added."""

    def __init__(self, path_file: str | Path) -> None:
        self.path_file = Path(path_file)

    # ========================================================
    # Reading
    # ========================================================

    def process_file_to_xarray(self) -> xr.Dataset:

        match = FILENAME_RE.search(self.path_file.name)
        if match is None:
            raise ValueError(f"Unexpected file name pattern: {self.path_file.name}")

        run = pd.Timestamp(match.group(1))
        fh = int(match.group(2))
        valid = run + pd.Timedelta(hours=fh)

        ds = xr.open_dataset(self.path_file, engine="netcdf4")
        ds = ds.rename({"latitude": "lat", "longitude": "lon"})

        if "time" in ds.dims:
            ds = ds.isel(time=0, drop=True)
        ds = ds.drop_vars(["reference_time", "step"], errors="ignore")

        ds = ds.expand_dims(time=[valid])
        ds.attrs["run_time"] = str(run)
        ds.attrs["forecast_hour"] = fh
        return ds

    # ========================================================
    # Formulas
    # ========================================================

    @staticmethod
    def wind_shear(u_surf, v_surf, u_upper, v_upper):
        return np.sqrt(((u_surf - u_upper) ** 2) + (v_surf - v_upper) ** 2)

    def dls(self, ds: xr.Dataset) -> xr.Dataset:
        """Mid-level wind shear (surface -> 450 hPa)."""
        ds["u_mean_sup"] = (ds["u10"] + ds["u_975"]) / 2
        ds["v_mean_sup"] = (ds["v10"] + ds["v_975"]) / 2
        ds["dls"] = self.wind_shear(
            ds["u_mean_sup"], ds["v_mean_sup"], ds["u_450"], ds["v_450"],
        )
        return ds[["dls"]]

    @staticmethod
    def thermal_gradient(ds: xr.Dataset) -> xr.Dataset:

        temp_850 = ds["t850"]
        lat = ds["lat"].values
        lon = ds["lon"].values

        lat_rad = np.deg2rad(lat)
        dlat = np.gradient(lat) * 111.32e3                     # 1 deg ~ 111.32 km
        dlon = np.gradient(lon) * (111.32e3 * np.cos(lat_rad.mean()))

        dT_dlat, dT_dlon = np.gradient(temp_850.values, axis=(-2, -1))
        dT_dlat = dT_dlat / dlat[:, None]
        dT_dlon = dT_dlon / dlon[None, :]

        return xr.Dataset(
            {
                "dT_dlat": (["time", "lat", "lon"], dT_dlat, temp_850.attrs),
                "dT_dlon": (["time", "lat", "lon"], dT_dlon, temp_850.attrs),
                "grad_T": (
                    ["time", "lat", "lon"],
                    np.sqrt(dT_dlat ** 2 + dT_dlon ** 2),
                    temp_850.attrs,
                ),
            },
            coords={"time": ds["time"], "lat": ds["lat"], "lon": ds["lon"]},
        )

    @staticmethod
    def calculate_dew_point(temp, rh):
        a = 17.27
        b = 237.7                                              # deg C
        rh_safe = np.maximum(rh, 1e-6)
        alpha = (a * temp) / (b + temp) + np.log(rh_safe / 100.0)
        return (b * alpha) / (a - alpha)

    @staticmethod
    def calculate_tt(t850, t500, td850):
        return (t850 - t500) + (td850 - t500)

    @staticmethod
    def calculate_ki(t850, t500, td850, t700, td700):
        return (t850 - t500) + td850 - (t700 - td700)

    @staticmethod
    def calculate_wind_divergence(ds: xr.Dataset, u_var: str, v_var: str):

        u = ds[u_var].values
        v = ds[v_var].values

        lat_rad = np.radians(ds["lat"].values)
        lon_rad = np.radians(ds["lon"].values)

        dlat = np.gradient(lat_rad)[:, None]
        dlon = np.gradient(lon_rad)[None, :]

        radius = 6371000.0

        du_dx = np.gradient(u, axis=-1) / (radius * np.cos(lat_rad)[:, None] * dlon)
        dv_dy = np.gradient(v, axis=-2) / (radius * dlat)

        return xr.DataArray(
            du_dx + dv_dy,
            coords=ds[u_var].coords,
            dims=ds[u_var].dims,
        )

    # ========================================================
    # Per-file pipeline
    # ========================================================

    def compute(self) -> xr.Dataset:

        ds = self.process_file_to_xarray()

        t850 = ds["t850"] - 273.15
        t500 = ds["t500"] - 273.15
        t700 = ds["t700"] - 273.15

        td850 = self.calculate_dew_point(t850, ds["ur850"])
        td700 = self.calculate_dew_point(t700, ds["ur700"])

        ds["TT"] = self.calculate_tt(t850, t500, td850)
        ds["KI"] = self.calculate_ki(t850, t500, td850, t700, td700)
        ds["div850"] = self.calculate_wind_divergence(ds, "u_850", "v_850")
        ds["div250"] = self.calculate_wind_divergence(ds, "u_250", "v_250")
        ds["lls"] = self.wind_shear(ds["u10"], ds["v10"], ds["u_1000"], ds["v_1000"])
        ds["dls"] = self.dls(ds)["dls"]

        grad = self.thermal_gradient(ds)
        for var in grad.data_vars:
            ds[var] = grad[var]

        return ds


def compute_parameters_for_files(
    gfs_files,
    output_dir: Path,
    force: bool = False,
) -> list[Path]:
    """Compute the indices for each GFS NetCDF and save them in output_dir."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    produced: list[Path] = []

    for item in gfs_files:

        output_file = output_dir / item.path.name

        if output_file.exists() and not force:
            log.info("Parameters already computed: %s", output_file.name)
            produced.append(output_file)
            continue

        try:
            ds = AtmosphericParameters(item.path).compute()

            ds = ds.assign_coords(
                run_time=np.datetime64(item.run_date.strftime("%Y-%m-%dT%H:%M"), "ns"),
                valid_time=np.datetime64(
                    item.valid_time.strftime("%Y-%m-%dT%H:%M"), "ns"
                ),
            )
            ds.attrs["forecast_hour"] = item.forecast_hour

            encoding = {
                var: {"zlib": True, "complevel": 4} for var in ds.data_vars
            }
            tmp = output_file.with_suffix(".tmp.nc")
            ds.to_netcdf(tmp, encoding=encoding)
            ds.close()
            tmp.replace(output_file)

            produced.append(output_file)
            log.info("Parameters saved: %s", output_file.name)

        except Exception as error:                              # noqa: BLE001
            log.exception("Failed to compute parameters for %s: %s", item.path.name, error)

    return sorted(produced)
