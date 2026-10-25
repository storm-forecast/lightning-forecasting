"""ECMWF IFS open data (oper, 00 UTC) delivered as the fields the GFS path produces.

The point of this module is that nothing downstream of it has to know ECMWF exists. It writes the
same NetCDF, with the same variable names and units, as `gfs.py` -- so `parameters.py`,
`dataset.py` and `predict.py` run unchanged on either source.

Same cycle stitching as the GFS path:

    two days ago (00z) -> leads 6, 12, 18, 24
    yesterday    (00z) -> leads 6, 12, 18, 24
    today        (00z) -> leads 6 ... horizon_hours

Three things in the open data do not line up with GFS, and this module absorbs all three:

1. 975 and 450 hPa are not published. The level list is 10, 50, 100, 150, 200, 250, 300, 400, 500,
   600, 700, 850, 925, 1000 -- so `dls` (surface -> 450) and the 975 hPa wind it averages are
   interpolated in log-pressure from the levels that bracket them. For wind over 25-50 hPa that is
   a smaller error than the grid itself.

2. There is no surface-based CAPE in the open data, only `mucape` (most-unstable) -- and not in the
   ensemble stream either, so switching streams would not recover it; the true `cape` (paramId 59)
   sits in ECMWF's licensed catalogue. The decision is to feed MUCAPE to the models as their CAPE
   input. It is the one substitution that changes the physics of the input rather than its
   packaging: MUCAPE >= surface-based CAPE always, and the gap is widest under an inversion, where
   the most unstable parcel is not the surface one. The models were trained on the GFS
   surface-based field, so the IFS lightning output carries that bias until it is measured against
   the GFS one -- which is what the dated release copies exist for. It is a single entry in
   VARIABLES, so it is also the one line to change should that comparison call for deriving a
   surface-based value from the profiles instead.

3. `tp` accumulates from the start of the cycle and is in metres, where GFS gives 6 h buckets in
   kg m-2. Consecutive steps are differenced and scaled, which is why the steps of a cycle are
   processed in order rather than as they arrive.

Downloads use the byte ranges in the `.index` that sits beside each GRIB, so only the needed
messages are fetched -- a few MB instead of the whole global file. The messages are identified by
their position in that index rather than by eccodes shortName: the index is ECMWF's own statement
of what each message is, and concatenating ranges in offset order keeps message order in step with
it. One less table to be wrong about.
"""

from __future__ import annotations

import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import requests
from netCDF4 import Dataset, date2num

from app.core.gfs import GFSFile

log = logging.getLogger(__name__)

BASE_URL = "https://data.ecmwf.int/forecasts"
TIME_UNITS = "hours since 1970-01-01 00:00:00"

# (output_name, open-data param, level in hPa or None for single level).
# The output names are the GFS ones on purpose; see the module docstring.
VARIABLES: list[tuple[str, str, int | None]] = [
    ("u10", "10u", None),
    ("v10", "10v", None),
    ("prmsl", "msl", None),
    ("cape", "mucape", None),          # most-unstable, used as the CAPE input -- see docstring
    ("tp", "tp", None),                # cumulative, in m -- differenced below
    ("t2m", "2t", None),
    ("td2m", "2d", None),
    ("ur600", "r", 600),
    ("ur700", "r", 700),
    ("ur850", "r", 850),
    ("ur500", "r", 500),
    ("u_1000", "u", 1000),
    ("v_1000", "v", 1000),
    ("u_850", "u", 850),
    ("v_850", "v", 850),
    ("u_250", "u", 250),
    ("v_250", "v", 250),
    ("w500", "w", 500),
    ("t850", "t", 850),
    ("t500", "t", 500),
    ("t700", "t", 700),
]

# (output_name, param, wanted level, lower level, upper level) -- levels absent from the open data,
# interpolated from the two that bracket them.
INTERPOLATED: list[tuple[str, str, int, int, int]] = [
    ("u_975", "u", 975, 1000, 925),
    ("v_975", "v", 975, 1000, 925),
    ("u_450", "u", 450, 500, 400),
    ("v_450", "v", 450, 500, 400),
]

ACCUM_VAR = "tp"
# m -> kg m-2, which is what the GFS files carry and what build_precipitation_dataset expects.
ACCUM_SCALE = 1000.0


def _log_p_weight(target: int, lower: int, upper: int) -> float:
    """Weight of `upper` when interpolating to `target` linearly in ln(p)."""
    return float(np.log(target / lower) / np.log(upper / lower))


class ECMWFDownloader:
    """Same constructor and same `run()` contract as GFSDownloader."""

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

        self.variables = ([v for v in VARIABLES if v[0] in only] if only else list(VARIABLES))
        if not self.variables:
            raise ValueError(f"No known variable among {only}")

        # Interpolated levels cost two downloads each and are only worth fetching when the caller
        # actually wants them -- the worldwide precipitation pass, for one, asks for `tp` alone.
        self.interpolated = (
            [v for v in INTERPOLATED if v[0] in only] if only else list(INTERPOLATED)
        )

        # What to pull out of the index: (param, level) -> the output names it feeds. A bracketing
        # level can serve two outputs (u_975 and the 1000 hPa wind both want u at 1000), so the
        # mapping is one key to many names.
        self.wanted: dict[tuple[str, int | None], list[str]] = {}
        for name, param, level in self.variables:
            self.wanted.setdefault((param, level), []).append(name)
        for _name, param, _target, lower, upper in self.interpolated:
            for level in (lower, upper):
                self.wanted.setdefault((param, level), [])

        self.output_base = Path(output_base)
        self.lon_range = lon_range
        self.lat_range = lat_range
        self.max_workers = max_workers
        self.cleanup_grib = cleanup_grib

        self.leads_past = [6, 12, 18, 24]
        self.leads_today = list(range(6, horizon_hours + 1, 6))

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
    # Paths in the bucket
    # ========================================================

    def _base_name(self, date: datetime, fh: int) -> str:
        return f"{date:%Y%m%d}000000-{fh}h-oper-fc"

    def _url(self, date: datetime, fh: int, suffix: str) -> str:
        return f"{BASE_URL}/{date:%Y%m%d}/00z/ifs/0p25/oper/{self._base_name(date, fh)}.{suffix}"

    # ========================================================
    # .index -> byte ranges
    # ========================================================

    def _entries(self, index_text: str) -> list[tuple[int, int, tuple[str, int | None]]]:
        """The wanted messages as (offset, length, key), in offset order."""

        found: list[tuple[int, int, tuple[str, int | None]]] = []

        for line in index_text.splitlines():
            if not line.strip():
                continue
            row = json.loads(line)

            levtype = row.get("levtype")
            if levtype == "sfc":
                key = (row.get("param"), None)
            elif levtype == "pl":
                key = (row.get("param"), int(row["levelist"]))
            else:
                # sol (soil) and anything else added later: never asked for, and its levelist
                # numbering would collide with pressure levels.
                continue

            if key in self.wanted:
                found.append((int(row["_offset"]), int(row["_length"]), key))

        found.sort(key=lambda item: item[0])

        missing = set(self.wanted) - {key for _o, _l, key in found}
        if missing:
            log.warning("Messages absent from the index: %s", sorted(map(str, missing)))

        return found

    @staticmethod
    def _ranges(entries) -> list[tuple[int, int]]:
        """Contiguous messages merged into one request each."""

        merged: list[list[int]] = []

        for offset, length, _key in entries:
            end = offset + length - 1
            if merged and offset == merged[-1][1] + 1:
                merged[-1][1] = end
            else:
                merged.append([offset, end])

        return [(start, end) for start, end in merged]

    # ========================================================
    # Download
    # ========================================================

    def download_one(self, date: datetime, fh: int):
        """Returns (path, entries) so `process` knows what each message is."""

        grib_dir = self.output_base / date.strftime("%Y") / date.strftime("%m") / f"{date:%Y%m%d}"
        grib_dir.mkdir(parents=True, exist_ok=True)

        target = grib_dir / f"{self._base_name(date, fh)}.grib2"

        try:
            index_resp = self.session.get(self._url(date, fh, "index"), timeout=60)

            if index_resp.status_code == 404:
                log.warning("Not published in the bucket yet: %s %dh", f"{date:%Y%m%d}", fh)
                return None

            index_resp.raise_for_status()
            entries = self._entries(index_resp.text)

            if not entries:
                log.error("No wanted message in the index for %s %dh", f"{date:%Y%m%d}", fh)
                return None

            # The index is fetched even when the GRIB is already on disk: it is a few kB and it is
            # what tells `process` which message is which.
            if target.exists() and target.stat().st_size > 0:
                log.info("GRIB already on disk: %s", target.name)
                return target, entries

            tmp = target.with_suffix(target.suffix + ".part")

            with open(tmp, "wb") as handle:
                for start, end in self._ranges(entries):
                    resp = self.session.get(
                        self._url(date, fh, "grib2"),
                        headers={"Range": f"bytes={start}-{end}"},
                        timeout=180,
                    )
                    resp.raise_for_status()
                    handle.write(resp.content)

            os.replace(tmp, target)
            log.info(
                "Download OK (%.1f MB): %s %dh",
                target.stat().st_size / 1e6, f"{date:%Y%m%d}", fh,
            )
            return target, entries

        except requests.RequestException as error:
            log.error("Download failed for %s %dh: %s", f"{date:%Y%m%d}", fh, error)
            return None

    def download(self, date: datetime, leads: list[int]) -> dict[int, tuple]:

        out: dict[int, tuple] = {}

        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            futures = {pool.submit(self.download_one, date, fh): fh for fh in leads}
            for future in as_completed(futures):
                result = future.result()
                if result is not None:
                    out[futures[future]] = result

        return out

    # ========================================================
    # GRIB -> arrays
    # ========================================================

    def process(self, grib_file: Path, entries) -> tuple[np.ndarray, np.ndarray, dict]:
        import pygrib

        log.info("Processing GRIB: %s", grib_file.name)

        grbs = pygrib.open(str(grib_file))

        try:
            messages = list(grbs)

            # The concatenation was written in offset order, so message i is entries[i]. If the
            # counts disagree something is wrong with that assumption and guessing would silently
            # mislabel fields -- which is worse than losing the lead time.
            if len(messages) != len(entries):
                raise ValueError(
                    f"{len(messages)} messages for {len(entries)} index entries"
                )

            latitude, longitude = messages[0].latlons()

            lat = latitude[:, 0].copy()
            lon = longitude[0, :].copy()
            lon[lon > 180] -= 360

            lat_idx = np.where((lat >= self.lat_range[0]) & (lat <= self.lat_range[1]))[0]
            lon_idx = np.where((lon >= self.lon_range[0]) & (lon <= self.lon_range[1]))[0]

            lat_min, lat_max = lat_idx[0], lat_idx[-1]
            lon_min, lon_max = lon_idx[0], lon_idx[-1]

            raw: dict[tuple[str, int | None], np.ndarray] = {}

            for message, (_offset, _length, key) in zip(messages, entries):
                raw[key] = message.values[lat_min:lat_max + 1, lon_min:lon_max + 1]

        finally:
            grbs.close()

        data: dict[str, np.ndarray] = {}

        for name, param, level in self.variables:
            values = raw.get((param, level))
            if values is None:
                log.warning("Variable %s was not extracted", name)
                continue
            data[name] = values

        for name, param, target, lower, upper in self.interpolated:
            below = raw.get((param, lower))
            above = raw.get((param, upper))
            if below is None or above is None:
                log.warning("Variable %s needs %d and %d hPa, which are missing", name, lower, upper)
                continue
            weight = _log_p_weight(target, lower, upper)
            data[name] = below + weight * (above - below)

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
                if name == ACCUM_VAR:
                    var.long_name = "precipitation accumulated over the previous 6 h"
                    var.units = "kg m-2"
                var[0, :, :] = values

            nc.title = "ECMWF IFS 0.25 deg (oper) - South America subset"
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
                "ECMWF %s 00z - %d lead times (%dh to %dh)",
                date_str, len(leads), leads[0], leads[-1],
            )

            output_dir = self.output_base / date.strftime("%Y") / date.strftime("%m")
            output_dir.mkdir(parents=True, exist_ok=True)

            pending = []
            for fh in leads:
                nc_file = output_dir / f"ifs.{date_str}.00z.f{fh:03d}.nc"
                if nc_file.exists() and nc_file.stat().st_size > 0:
                    produced.append(GFSFile(date, fh, nc_file))
                else:
                    pending.append(fh)

            if not pending:
                log.info("Every NetCDF for %s already exists.", date_str)
                continue

            # `tp` is differenced against the step before, so a step whose predecessor is not in
            # this batch needs it downloaded too, even though it is already converted.
            needed = set(pending)
            if ACCUM_VAR in {v[0] for v in self.variables}:
                needed |= {fh - 6 for fh in pending if fh - 6 >= 6}

            downloaded = self.download(date, sorted(needed))

            # In step order, so the previous accumulation is in hand.
            previous: np.ndarray | None = None
            previous_fh: int | None = None

            for fh in sorted(downloaded):

                grib_file, entries = downloaded[fh]
                output_file = output_dir / f"ifs.{date_str}.00z.f{fh:03d}.nc"

                try:
                    lat, lon, data = self.process(grib_file, entries)

                    if ACCUM_VAR in {v[0] for v in self.variables}:
                        if ACCUM_VAR not in data:
                            raise ValueError("file has no precipitation (tp)")

                        cumulative = data[ACCUM_VAR]

                        # A step fetched only to be the predecessor of a pending one is never
                        # saved, so its own bucket is never needed -- only its running total, as
                        # the baseline the next step subtracts.
                        if fh in pending:
                            if fh == 6:
                                # Nothing has fallen before the cycle starts, so the first step is
                                # already the 6 h bucket.
                                bucket = cumulative
                            elif previous is not None and previous_fh == fh - 6:
                                bucket = cumulative - previous
                            else:
                                raise ValueError(
                                    f"cannot difference tp at {fh}h without {fh - 6}h"
                                )
                            data[ACCUM_VAR] = np.maximum(bucket, 0.0) * ACCUM_SCALE

                        previous, previous_fh = cumulative, fh

                    if fh not in pending:
                        # Downloaded only to serve as the predecessor of a pending step.
                        continue

                    self.save(output_file, lat, lon, data, date, fh)
                    produced.append(GFSFile(date, fh, output_file))
                    log.info("NetCDF saved: %s", output_file.name)

                except Exception as error:                   # noqa: BLE001
                    log.error("Failed to convert %s: %s", grib_file.name, error)
                    output_file.unlink(missing_ok=True)
                    if fh in pending:
                        missing.append((date, fh))
                    previous, previous_fh = None, None
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

        try:
            resp = self.session.head(
                self._url(self.run_date, self.leads_today[-1], "index"), timeout=30,
            )
            return resp.status_code == 200
        except requests.RequestException:
            return False
