"""Publish the newest run's forecast to Cloudflare R2, where the map reads it.

Why R2 and not this service's own API: the API has to be running to serve a file, and a web service
that runs a ConvLSTM needs memory it only has on a paid plan. A bucket serves the same bytes for free,
and the map only ever needed a URL whose Last-Modified changes when a new run lands -- which is exactly
what an object store gives.

The key is fixed (`lightning/gfs_prediction_ALL_regions_suavisado.nc`), overwritten every run, because
the consumer is configured with one URL and cannot be told a new run id each day.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import boto3
from botocore.config import Config

from app.config import settings
from app.core import state

KEY = "lightning/gfs_prediction_ALL_regions_suavisado.nc"
PRECIP_KEY = "lightning/gfs_precipitation_suavisado.nc"


def r2_client():
    return boto3.client(
        "s3",
        endpoint_url=f"https://{os.environ['R2_ACCOUNT_ID']}.r2.cloudflarestorage.com",
        aws_access_key_id=os.environ["R2_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["R2_SECRET_ACCESS_KEY"],
        # R2 rejects the default checksum headers of recent boto3 versions on some operations.
        config=Config(request_checksum_calculation="when_required",
                      response_checksum_validation="when_required"),
    )


def main() -> int:
    run_id = state.get_latest_run_id(settings.products_dir)
    if run_id is None:
        print("No run on disk to publish.", file=sys.stderr)
        return 1

    base = state.run_dir(settings.products_dir, run_id)
    bucket = os.environ["R2_BUCKET"]
    s3 = r2_client()

    enviados = 0
    for relativo, key in (
        (f"{state.LIGHTNING_SUBDIR}/gfs_prediction_ALL_regions_suavisado.nc", KEY),
        (f"{state.PRECIP_SUBDIR}/gfs_precipitation_suavisado.nc", PRECIP_KEY),
    ):
        caminho = base / relativo
        if not caminho.is_file():
            print(f"missing, skipped: {relativo}")
            continue
        # no-cache so the consumer's conditional request reaches the object instead of a CDN copy:
        # it polls every ten minutes and must see a new run the moment it lands.
        s3.upload_file(str(caminho), bucket, key, ExtraArgs={
            "ContentType": "application/x-netcdf",
            "CacheControl": "no-cache",
            "Metadata": {"run-id": run_id},
        })
        print(f"{caminho.name} ({caminho.stat().st_size/1e6:.1f} MB) -> s3://{bucket}/{key}")
        enviados += 1

    if not enviados:
        print("Nothing published.", file=sys.stderr)
        return 1
    print(f"\nRun {run_id} published.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
