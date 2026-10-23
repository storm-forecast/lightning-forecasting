"""Smoke test: exercises the pipeline with no network, GRIB or models.

Fabricates NetCDFs identical in structure to what the GFS download writes,
then runs steps 3 through 7 (parameters, dataset, merge, smoothing,
precipitation). The prediction itself is replaced by synthetic fields, since
the real models live on Drive.

    python scripts/smoke_test.py
"""

from __future__ import annotations

import sys
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

LAT = np.arange(-50.0, 15.01, 0.25)
LON = np.arange(-85.0, -30.01, 0.25)

failures: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    if condition:
        print(f"  [ok]   {name}")
    else:
        print(f"  [FAIL] {name} {detail}")
        failures.append(name)


def fabricate_gfs(base: Path, run: datetime, fh: int):
    """Write a NetCDF with the same structure as GFSDownloader.save."""
    from app.core.gfs import VARIABLES, GFSDownloader, GFSFile

    rng = np.random.default_rng(fh + run.day * 100)
    shape = (len(LAT), len(LON))

    data = {}
    for name, _short, _level, _pattern in VARIABLES:
        if name.startswith(("t2m", "t850", "t500", "t700")):
            field = 273.15 + rng.uniform(-20, 30, shape)
        elif name.startswith("ur") or name == "td2m":
            field = rng.uniform(10, 100, shape)
        elif name == "prmsl":
            field = rng.uniform(98000, 103000, shape)
        elif name == "cape":
            field = rng.uniform(0, 3000, shape)
        elif name == "tp":
            field = np.maximum(rng.normal(1.0, 4.0, shape), 0.0)
        else:
            field = rng.uniform(-30, 30, shape)
        data[name] = field.astype("float32")

    out_dir = base / run.strftime("%Y") / run.strftime("%m")
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / f"gfs.{run:%Y%m%d}.00z.f{fh:03d}.nc"

    GFSDownloader.save(target, LAT, LON, data, run, fh)
    return GFSFile(run, fh, target)


def main() -> int:

    import xarray as xr

    from app.core.dataset import build_input_dataset
    from app.core.parameters import compute_parameters_for_files
    from app.core.pipeline import merge_regions
    from app.core.precipitation import (
        VAR_6H, VAR_24H, VAR_TOTAL, build_precipitation_dataset,
    )
    from app.core.predict import create_rolling_window, forecast_to_dataarray
    from app.core.regions import MODEL_VARIABLES, VAR_DETER, VAR_PROB
    from app.core.smoothing import smooth_dataset

    tmp = Path(tempfile.mkdtemp(prefix="smoke-"))
    today = datetime(2026, 10, 23, tzinfo=timezone.utc)

    # ---------- 1. GFS NetCDFs ----------
    print("\n1. Fabricating the GFS NetCDFs (same structure as the download)")
    files = []
    for days_back, leads in ((2, [6, 12, 18, 24]), (1, [6, 12, 18, 24]),
                             (0, [6, 12, 18, 24, 30, 36, 42, 48])):
        run = today - timedelta(days=days_back)
        for fh in leads:
            files.append(fabricate_gfs(tmp / "gfs", run, fh))
    check("16 lead times generated", len(files) == 16, f"-> {len(files)}")

    # ---------- 2. Atmospheric parameters ----------
    print("\n2. Computing the atmospheric parameters")
    param_files = compute_parameters_for_files(files, tmp / "parameters")
    check("one file per lead time", len(param_files) == 16, f"-> {len(param_files)}")

    with xr.open_dataset(param_files[0]) as ds0:
        for var in ("TT", "KI", "div850", "div250", "lls", "dls", "grad_T"):
            check(f"index {var} computed",
                  var in ds0.data_vars and np.isfinite(ds0[var]).any())

    # ---------- 3. Input dataset ----------
    print("\n3. Building the input dataset")
    ds_synoptic, ds_precip = build_input_dataset(param_files)
    check("15 input variables",
          list(ds_synoptic.data_vars) == MODEL_VARIABLES,
          f"-> {list(ds_synoptic.data_vars)}")
    check("16 continuous valid times", ds_synoptic.sizes["time"] == 16,
          f"-> {ds_synoptic.sizes['time']}")

    steps = np.diff(ds_synoptic["time"].values) / np.timedelta64(1, "h")
    check("6 h spacing with no repeats", bool((steps == 6).all()),
          f"-> {sorted(set(steps.tolist()))}")
    check("precipitation preserved", "tp" in ds_precip.data_vars)

    # ---------- 4. Time windows ----------
    print("\n4. Checking the model windows")
    stacked = np.stack([ds_synoptic[v].values for v in MODEL_VARIABLES], axis=-1)
    for window, expected in ((6, 10), (5, 11)):
        rolling = create_rolling_window(stacked, window)
        check(f"window {window} -> {expected} predictions",
              rolling.shape[0] == expected, f"-> {rolling.shape[0]}")

    # ---------- 5. Synthetic prediction and time alignment ----------
    print("\n5. Synthetic per-region prediction and time alignment")
    regions = []
    for i, (lon0, lon1, lat0, lat1) in enumerate(
        [(-64, -50, -36, -20), (-60, -44, -35, -19), (-70, -52, -14, -2)]
    ):
        sub = ds_synoptic.sel(lon=slice(lon0, lon1), lat=slice(lat0, lat1))
        shape = (10, sub.sizes["lat"], sub.sizes["lon"])
        rng = np.random.default_rng(i)

        prob = forecast_to_dataarray(
            rng.uniform(0, 1, shape).astype("float32"), sub, VAR_PROB,
        ).to_dataset()
        prob[VAR_PROB] = prob[VAR_PROB].where(prob[VAR_PROB] > 0.15, 0.0)

        deter = forecast_to_dataarray(
            rng.uniform(0, 400, shape).astype("float32"), sub, VAR_DETER,
        ).to_dataset()

        merged_region = xr.merge([prob, deter], join="exact")
        merged_region[VAR_DETER] = merged_region[VAR_DETER].where(
            merged_region[VAR_PROB] != 0, 0,
        )
        merged_region[VAR_DETER] = merged_region[VAR_DETER].astype(np.int32)
        regions.append(merged_region)

    check("predictions aligned to the last 10 valid times",
          list(regions[0]["time"].values) == list(ds_synoptic["time"].values[-10:]))

    # ---------- 6. Region merge ----------
    print("\n6. Merging the regions by maximum")
    merged = merge_regions(regions)
    check("no NaN in the probability field", int(merged[VAR_PROB].isnull().sum()) == 0)
    check("no NaN in the deterministic field", int(merged[VAR_DETER].isnull().sum()) == 0)
    check("probability zeroed where there is no lightning",
          bool((merged[VAR_PROB].where(merged[VAR_DETER] == 0, 0) == 0).all()))
    check("union grid covers every region",
          merged.sizes["lat"] >= max(r.sizes["lat"] for r in regions))
    check("threshold respected",
          bool(((merged[VAR_PROB] == 0) | (merged[VAR_PROB] > 0.15)).all()))

    # ---------- 7. Smoothing ----------
    print("\n7. Smoothing the fields")
    smoothed = smooth_dataset(
        merged,
        {
            VAR_DETER: dict(upscale=2, sigma=1.2, despeckle=True,
                            vmin=0.0, vmax=None, remask=True),
            VAR_PROB: dict(upscale=2, sigma=1.2, despeckle=True,
                           vmin=0.0, vmax=1.0, remask=True),
        },
        max_side=2400,
    )
    check("grid refined", smoothed.sizes["lat"] > merged.sizes["lat"])
    check("valid times preserved", smoothed.sizes["time"] == merged.sizes["time"])
    check("probability still within [0, 1]",
          float(smoothed[VAR_PROB].min()) >= 0 and float(smoothed[VAR_PROB].max()) <= 1.0)
    check("deterministic field non-negative", float(smoothed[VAR_DETER].min()) >= 0)

    # ---------- 8. Precipitation ----------
    print("\n8. Generating the precipitation")
    rain = build_precipitation_dataset(ds_precip, today, accum_24h=True)
    check("three rainfall fields",
          all(v in rain.data_vars for v in (VAR_6H, VAR_24H, VAR_TOTAL)))
    check("6 h accumulation non-negative", float(rain[VAR_6H].min()) >= 0)
    check("24 h >= 6 h where the window is complete",
          bool((rain[VAR_24H].isel(time=slice(3, None))
                >= rain[VAR_6H].isel(time=slice(3, None)) - 1e-3).all()))
    check("leading 24 h window marked as NaN",
          bool(rain[VAR_24H].isel(time=0).isnull().all()))
    check("total equals the sum of the future valid times",
          np.allclose(
              rain[VAR_TOTAL].values,
              rain[VAR_6H].isel(time=slice(-8, None)).sum("time").values,
              rtol=1e-4,
          ))

    # ---------- result ----------
    print("\n" + "=" * 58)
    if failures:
        print(f"FAILED: {len(failures)} check(s) -> {failures}")
        return 1
    print("Every check passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
