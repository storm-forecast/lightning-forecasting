"""GFS 0.25 deg download (00 UTC cycle) as one continuous time series.

Same logic as the original notebook:

    two days ago (00z) -> leads 6, 12, 18, 24
    yesterday    (00z) -> leads 6, 12, 18, 24
    today        (00z) -> leads 6 ... horizon_hours

Each cycle covers only the following 24 h, so the next cycle picks up where
the previous one stopped: no duplicated valid times, and f000 (the analysis,
which carries no precipitation) is never used.

Downloads use byte ranges driven by the .idx index of NOAA's public bucket,
so only the required variables are fetched (a few MB per file instead of
roughly 500 MB).
"""

from __future__ import annotations

import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import requests
from netCDF4 import Dataset, date2num

log = logging.getLogger(__name__)

BASE_URL = "https://noaa-gfs-bdp-pds.s3.amazonaws.com"
TIME_UNITS = "hours since 1970-01-01 00:00:00"

# (output_name, pygrib_shortName, level, idx_pattern)
VARIABLES: list[tuple[str, str, int | None, str]] = [
    ("u10", "10u", None, ":UGRD:10 m above ground:"),
    ("v10", "10v", None, ":VGRD:10 m above ground:"),
    ("prmsl", "prmsl", None, ":PRMSL:mean sea level:"),
    ("cape", "cape", None, ":CAPE:surface:"),
    ("tp", "tp", None, ":APCP:surface:"),
    ("t2m", "2t", None, ":TMP:2 m above ground:"),
    ("td2m", "2d", None, ":DPT:2 m above ground:"),
    ("ur600", "r", 600, ":RH:600 mb:"),
    ("u_975", "u", 975, ":UGRD:975 mb:"),
    ("v_975", "v", 975, ":VGRD:975 mb:"),
    ("u_1000", "u", 1000, ":UGRD:1000 mb:"),
    ("v_1000", "v", 1000, ":VGRD:1000 mb:"),
    ("u_250", "u", 250, ":UGRD:250 mb:"),
    ("v_250", "v", 250, ":VGRD:250 mb:"),
    ("u_450", "u", 450, ":UGRD:450 mb:"),
    ("v_450", "v", 450, ":VGRD:450 mb:"),
    ("u_850", "u", 850, ":UGRD:850 mb:"),
    ("v_850", "v", 850, ":VGRD:850 mb:"),
    ("w500", "w", 500, ":VVEL:500 mb:"),
    ("t850", "t", 850, ":TMP:850 mb:"),
    ("t500", "t", 500, ":TMP:500 mb:"),
    ("t700", "t", 700, ":TMP:700 mb:"),
    ("ur850", "r", 850, ":RH:850 mb:"),
    ("ur500", "r", 500, ":RH:500 mb:"),
    ("ur700", "r", 700, ":RH:700 mb:"),
]


@dataclass(frozen=True)
class GFSFile:
    run_date: datetime
    forecast_hour: int
    path: Path

    @property
    def valid_time(self) -> datetime:
        return self.run_date + timedelta(hours=self.forecast_hour)


class GFSDownloader:

    def __init__(
        self,
        output_base: Path,
        reference_time: datetime | None = None,
        horizon_hours: int = 48,
        lon_range: tuple[float, float] = (-85.0, -30.0),
        lat_range: tuple[float, float] = (-50.0, 15.0),
        max_workers: int = 6,
        cleanup_grib: bool = True,
        only: list[str] | None = None,
    ) -> None:

        # `only` keeps just these variables, by name. The whole point of the byte-range download is to
        # fetch only what is needed, and precipitation on its own needs one field out of fifteen --
        # which is what makes a global pass affordable where a global pass of everything would not be.
        self.variables = ([v for v in VARIABLES if v[0] in only] if only else list(VARIABLES))
        if not self.variables:
            raise ValueError(f"No known variable among {only}")

        self.output_base = Path(output_base)
        self.lon_range = lon_range
        self.lat_range = lat_range
        self.max_workers = max_workers
        self.cleanup_grib = cleanup_grib

        self.leads_past = [6, 12, 18, 24]
        self.leads_today = list(range(6, horizon_hours + 1, 6))

        self.idx_patterns = [v[3] for v in self.variables]

        if reference_time is None:
            reference_time = datetime.now(timezone.utc)
        if reference_time.tzinfo is None:
            reference_time = reference_time.replace(tzinfo=timezone.utc)
        else:
            reference_time = reference_time.astimezone(timezone.utc)

        today = datetime(
            reference_time.year, reference_time.month, reference_time.day,
            tzinfo=timezone.utc,
        )

        self.reference_time = reference_time
        self.run_date = today
        self.dates = [today - timedelta(days=2), today - timedelta(days=1), today]

        self.leads_by_date = {
            self.dates[0]: self.leads_past,
            self.dates[1]: self.leads_past,
            self.dates[2]: self.leads_today,
        }

        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "lightning-prediction/1.0"})

    # ========================================================
    # .idx index -> byte ranges
    # ========================================================

    def _byte_ranges(self, idx_text: str) -> list[list[int | None]]:

        entries: list[tuple[int, str]] = []

        for line in idx_text.splitlines():
            if not line.strip():
                continue
            parts = line.split(":")
            entries.append((int(parts[1]), line))

        entries.sort(key=lambda e: e[0])

        ranges: list[list[int | None]] = []

        for i, (start, line) in enumerate(entries):
            if any(p in line for p in self.idx_patterns):
                end = entries[i + 1][0] - 1 if i + 1 < len(entries) else None
                ranges.append([start, end])

        missing = [
            p for p in self.idx_patterns
            if not any(p in line for _, line in entries)
        ]
        if missing:
            log.warning("Variables missing from the idx: %s", missing)

        ranges.sort(key=lambda r: r[0])

        merged: list[list[int | None]] = []
        for item in ranges:
            if merged and merged[-1][1] is not None and item[0] == merged[-1][1] + 1:
                merged[-1][1] = item[1]
            else:
                merged.append(item)

        return merged

    # ========================================================
    # Download
    # ========================================================

    def download_one(self, date: datetime, fh: int) -> Path | None:

        date_str = date.strftime("%Y%m%d")
        fh_str = f"{fh:03d}"

        grib_dir = self.output_base / date.strftime("%Y") / date.strftime("%m") / date_str
        grib_dir.mkdir(parents=True, exist_ok=True)

        target = grib_dir / f"gfs.t00z.pgrb2.0p25.f{fh_str}"

        if target.exists() and target.stat().st_size > 0:
            log.info("GRIB already on disk: %s", target.name)
            return target

        grib_url = (
            f"{BASE_URL}/gfs.{date_str}/00/atmos/gfs.t00z.pgrb2.0p25.f{fh_str}"
        )

        try:
            idx_resp = self.session.get(grib_url + ".idx", timeout=60)

            if idx_resp.status_code == 404:
                log.warning("Not published in the bucket yet: %s f%s", date_str, fh_str)
                return None

            idx_resp.raise_for_status()
            ranges = self._byte_ranges(idx_resp.text)

            if not ranges:
                log.error("No variable found in the idx for %s f%s", date_str, fh_str)
                return None

            tmp = target.with_suffix(target.suffix + ".part")

            with open(tmp, "wb") as handle:
                for start, end in ranges:
                    rng = f"bytes={start}-" if end is None else f"bytes={start}-{end}"
                    resp = self.session.get(
                        grib_url, headers={"Range": rng}, timeout=180,
                    )
                    resp.raise_for_status()
                    handle.write(resp.content)

            os.replace(tmp, target)
            log.info(
                "Download OK (%.1f MB): %s f%s",
                target.stat().st_size / 1e6, date_str, fh_str,
            )
            return target

        except requests.RequestException as error:
            log.error("Download failed for %s f%s: %s", date_str, fh_str, error)
            return None

    def download(self, date: datetime, leads: list[int]) -> list[Path]:

        files: list[Path] = []

        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            futures = {pool.submit(self.download_one, date, fh): fh for fh in leads}
            for future in as_completed(futures):
                result = future.result()
                if result is not None:
                    files.append(result)

        return sorted(files)

    # ========================================================
    # GRIB -> arrays
    # ========================================================

    def process(self, grib_file: Path):
        # Imported here so the rest of the module stays usable in
        # environments without eccodes installed.
        import pygrib

        log.info("Processing GRIB: %s", grib_file.name)

        grbs = pygrib.open(str(grib_file))

        try:
            message = grbs.message(1)
            latitude, longitude = message.latlons()

            lat = latitude[:, 0].copy()
            lon = longitude[0, :].copy()
            lon[lon > 180] -= 360

            lat_idx = np.where(
                (lat >= self.lat_range[0]) & (lat <= self.lat_range[1])
            )[0]
            lon_idx = np.where(
                (lon >= self.lon_range[0]) & (lon <= self.lon_range[1])
            )[0]

            lat_min, lat_max = lat_idx[0], lat_idx[-1]
            lon_min, lon_max = lon_idx[0], lon_idx[-1]

            data: dict[str, np.ndarray] = {}

            for name, short_name, level, _pattern in self.variables:
                try:
                    if short_name == "tp":
                        # Some lead times carry two accumulations (the last
                        # 6 h and the total since the cycle start). Always keep
                        # the 6 h one so 'tp' means the same in every file.
                        msgs = grbs.select(shortName="tp")
                        six_hour = [
                            m for m in msgs
                            if int(m["endStep"]) - int(m["startStep"]) == 6
                        ]
                        message = (six_hour or msgs)[0]
                    elif level is not None:
                        message = grbs.select(shortName=short_name, level=level)[0]
                    else:
                        message = grbs.select(shortName=short_name)[0]

                    data[name] = message.values[
                        lat_min:lat_max + 1, lon_min:lon_max + 1
                    ]

                except (IndexError, KeyError, ValueError) as error:
                    log.warning("Variable %s was not extracted: %s", name, error)

        finally:
            grbs.close()

        return lat[lat_min:lat_max + 1], lon[lon_min:lon_max + 1], data

    # ========================================================
    # NetCDF
    # ========================================================

    @staticmethod
    def save(
        output_file: Path,
        lat: np.ndarray,
        lon: np.ndarray,
        data: dict[str, np.ndarray],
        run_date: datetime,
        fh: int,
    ) -> None:

        valid = run_date + timedelta(hours=fh)

        with Dataset(output_file, "w", format="NETCDF4") as nc:

            nc.createDimension("time", None)
            nc.createDimension("latitude", len(lat))
            nc.createDimension("longitude", len(lon))

            t = nc.createVariable("time", "f8", ("time",))
            t.units = TIME_UNITS
            t.calendar = "standard"
            t.standard_name = "time"
            t.long_name = "valid time"
            t[:] = [date2num(valid.replace(tzinfo=None), TIME_UNITS)]

            rt = nc.createVariable("reference_time", "f8")
            rt.units = TIME_UNITS
            rt.calendar = "standard"
            rt.standard_name = "forecast_reference_time"
            rt.assignValue(date2num(run_date.replace(tzinfo=None), TIME_UNITS))

            st = nc.createVariable("step", "i4")
            st.units = "hours"
            st.standard_name = "forecast_period"
            st.assignValue(fh)

            la = nc.createVariable("latitude", "f4", ("latitude",))
            la.units = "degrees_north"
            la[:] = lat

            lo = nc.createVariable("longitude", "f4", ("longitude",))
            lo.units = "degrees_east"
            lo[:] = lon

            for name, values in data.items():
                var = nc.createVariable(
                    name, "f4", ("time", "latitude", "longitude"),
                    zlib=True, complevel=4,
                )
                var.coordinates = "reference_time step"
                if name == "tp":
                    var.long_name = "precipitation accumulated over the previous 6 h"
                    var.units = "kg m-2"
                var[0, :, :] = values

            nc.title = "GFS 0.25 deg - South America subset"
            nc.run = f"{run_date:%Y-%m-%d} 00 UTC"
            nc.forecast_hour = fh

    # ========================================================
    # Execution
    # ========================================================

    def run(self) -> list[GFSFile]:
        """Download and convert everything, returning the available NetCDFs."""

        produced: list[GFSFile] = []
        missing: list[tuple[datetime, int]] = []

        for date in self.dates:

            date_str = date.strftime("%Y%m%d")
            leads = self.leads_by_date[date]

            log.info(
                "GFS %s 00z - %d lead times (%dh to %dh)",
                date_str, len(leads), leads[0], leads[-1],
            )

            output_dir = self.output_base / date.strftime("%Y") / date.strftime("%m")
            output_dir.mkdir(parents=True, exist_ok=True)

            # Lead times whose NetCDF already exists need no download.
            pending = []
            for fh in leads:
                nc_file = output_dir / f"gfs.{date_str}.00z.f{fh:03d}.nc"
                if nc_file.exists() and nc_file.stat().st_size > 0:
                    produced.append(GFSFile(date, fh, nc_file))
                else:
                    pending.append(fh)

            if not pending:
                log.info("Every NetCDF for %s already exists.", date_str)
                continue

            grib_files = self.download(date, pending)
            downloaded: set[int] = set()

            for grib_file in grib_files:

                fh_str = grib_file.name.split(".")[-1]        # "f030"
                fh = int(fh_str[1:])
                downloaded.add(fh)

                output_file = output_dir / f"gfs.{date_str}.00z.{fh_str}.nc"

                try:
                    lat, lon, data = self.process(grib_file)
                    if "tp" not in data:
                        raise ValueError("file has no precipitation (tp)")
                    self.save(output_file, lat, lon, data, date, fh)
                    produced.append(GFSFile(date, fh, output_file))
                    log.info("NetCDF saved: %s", output_file.name)
                except Exception as error:                   # noqa: BLE001
                    log.error("Failed to convert %s: %s", grib_file.name, error)
                    output_file.unlink(missing_ok=True)
                    missing.append((date, fh))
                finally:
                    if self.cleanup_grib:
                        grib_file.unlink(missing_ok=True)

            missing += [(date, fh) for fh in pending if fh not in downloaded]

        if missing:
            log.warning(
                "Lead times with no NetCDF: %s",
                [f"{d:%Y%m%d} f{fh:03d}" for d, fh in sorted(missing)],
            )

        produced.sort(key=lambda f: (f.run_date, f.forecast_hour))
        return produced

    # ========================================================
    # Availability check
    # ========================================================

    def today_run_is_ready(self) -> bool:
        """True when today's last lead time is already in the bucket."""

        date_str = self.run_date.strftime("%Y%m%d")
        fh = self.leads_today[-1]
        url = (
            f"{BASE_URL}/gfs.{date_str}/00/atmos/"
            f"gfs.t00z.pgrb2.0p25.f{fh:03d}.idx"
        )
        try:
            resp = self.session.head(url, timeout=30)
            return resp.status_code == 200
        except requests.RequestException:
            return False
