"""Central configuration, read from environment variables.

On Render every key below is set under Environment -> Environment Variables
(or in the envVars block of render.yaml).
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ========================================================
    # Service
    # ========================================================

    app_name: str = "Lightning Prediction API"
    app_version: str = "1.0.0"
    log_level: str = "INFO"

    # Key required by the admin endpoints (POST /admin/*).
    # Left empty, the admin endpoints are disabled.
    api_key: str = ""

    # Origins allowed by CORS ("*" allows every origin).
    cors_origins: str = "*"

    # ========================================================
    # Directories
    #
    # On Render, mount a persistent disk at /var/data so that models
    # and products survive every deploy and restart.
    # ========================================================

    data_dir: Path = Path("/var/data")

    # ========================================================
    # Scheduling (daily run)
    # ========================================================

    enable_scheduler: bool = True

    # UTC time of the daily run. The 00z GFS cycle is usually complete
    # (through f048) in the NOAA bucket around 05:30-06:00 UTC.
    run_at_utc: str = "07:30"

    # When True, check on startup whether today's run already exists and
    # trigger the pipeline if it does not.
    run_on_startup: bool = True

    # ========================================================
    # GFS
    # ========================================================

    horizon_hours: int = 48
    gfs_max_workers: int = 6
    lon_min: float = -85.0
    lon_max: float = -30.0
    lat_min: float = -50.0
    lat_max: float = 15.0

    # A second, precipitation-only pass over the whole globe.
    #
    # It is a separate pass and not a wider `lon_min/lat_min` on purpose: the lightning models are
    # trained per region of South America and cannot be applied elsewhere, while the parameters feeding
    # them would be computed over the whole downloaded domain -- eighteen times the points, for fifteen
    # fields, to then be thrown away outside the regions. Precipitation needs one field and no model,
    # so asking for it separately is cheap where asking for everything is not.
    precip_global: bool = False
    precip_global_lon_min: float = -180.0
    precip_global_lon_max: float = 180.0
    precip_global_lat_min: float = -60.0   # below this is Antarctica, where GFS precipitation says little
    precip_global_lat_max: float = 75.0

    # Delete the raw GRIB files once converted to NetCDF.
    cleanup_grib: bool = True
    # Delete the run's working directory when finished.
    cleanup_workdir: bool = True

    # ========================================================
    # Models
    # ========================================================

    steps: str = "6H"
    classifier_window: int = 6
    regressor_window: int = 5
    prob_threshold: float = 0.15
    predict_batch_size: int = 8

    # Regions to process (empty = every region in regions.py).
    regions: str = ""

    # ========================================================
    # Google Drive (where the models live)
    #
    # Three ways to obtain the models, in priority order:
    #
    #   1. assets_local_dir  -> tree already present on disk
    #   2. gdrive_assets_zip_id -> a single .zip on Drive (simplest)
    #   3. gdrive_classification_folder_id + gdrive_regressor_folder_id
    #      -> recursive folder mirror through the Drive API
    # ========================================================

    assets_local_dir: str = ""

    gdrive_assets_zip_id: str = ""
    gdrive_classification_folder_id: str = ""
    gdrive_regressor_folder_id: str = ""
    gdrive_shapefiles_folder_id: str = ""

    # Either the service account JSON itself (paste the whole document)
    # or a path to a .json file holding that credential.
    gdrive_service_account_json: str = ""
    gdrive_service_account_file: str = ""

    # For publicly shared folders/files, with no service account.
    gdrive_use_gdown: bool = False

    # Revalidate the models against Drive on every run
    # (False = trust the local cache).
    refresh_assets_each_run: bool = False

    # ========================================================
    # Post-processing / web products
    # ========================================================

    smooth_upscale: int = 4
    smooth_sigma: float = 1.2

    # Cap on the longest side of the smoothed grid. Protects container
    # memory when smooth_upscale is high.
    smooth_max_side: int = 2400

    # Also smooth the precipitation fields (same recipe as lightning).
    smooth_precipitation: bool = True

    # Also build the 24 h accumulated precipitation field.
    precip_accum_24h: bool = True

    # How many past runs to keep on disk.
    keep_runs: int = 4

    # ========================================================
    # Derived values
    # ========================================================

    @field_validator("data_dir", mode="before")
    @classmethod
    def _expand(cls, value):
        return Path(os.path.expanduser(str(value)))

    @property
    def assets_dir(self) -> Path:
        if self.assets_local_dir:
            return Path(self.assets_local_dir)
        return self.data_dir / "assets"

    @property
    def work_dir(self) -> Path:
        return self.data_dir / "work"

    @property
    def products_dir(self) -> Path:
        return self.data_dir / "products"

    @property
    def state_file(self) -> Path:
        return self.data_dir / "state.json"

    @property
    def classification_root(self) -> Path:
        """Mirrors .../Doutorado/Model-Trained on Drive."""
        return self.assets_dir / "Model-Trained"

    @property
    def regressor_root(self) -> Path:
        """Mirrors .../Doutorado/Model-Trained-new on Drive."""
        return self.assets_dir / "Model-Trained-new"

    @property
    def shapefiles_root(self) -> Path:
        return self.assets_dir / "shapefiles_clusters_with_buffer"

    @property
    def region_list(self) -> list[str]:
        from app.core.regions import DEFAULT_REGIONS

        if not self.regions.strip():
            return list(DEFAULT_REGIONS)
        return [r.strip() for r in self.regions.split(",") if r.strip()]

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    def ensure_dirs(self) -> None:
        for path in (self.data_dir, self.assets_dir, self.work_dir, self.products_dir):
            path.mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
