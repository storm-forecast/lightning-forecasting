"""Regions (clusters) used by the models and their spatial extents.

Each region has its own classification and regression models. The spatial
subset for a region comes from the bounding box of its shapefile.

To avoid depending on GDAL/geopandas on the server, the bounding box is read
straight from the .shp header (bytes 36-68 of the ESRI specification), or
from a pre-computed JSON file at config/region_bounds.json.
"""

from __future__ import annotations

import json
import struct
from dataclasses import dataclass
from pathlib import Path

CONFIG_DIR = Path(__file__).resolve().parents[2] / "config"
BOUNDS_FILE = CONFIG_DIR / "region_bounds.json"


@dataclass(frozen=True)
class RegionMeta:
    name: str          # shapefile name, including the _buffered suffix
    label: str         # R1..R8
    color: str         # color used on reference maps
    title: str         # human-readable name, for the web page


REGION_META: dict[str, RegionMeta] = {
    "Nordeste_Brasileiro_buffered": RegionMeta(
        "Nordeste_Brasileiro_buffered", "R1", "darkred", "Northeast Brazil",
    ),
    "Sudeste_Brasileiro_Uruguai_Oeste_Argentino_buffered": RegionMeta(
        "Sudeste_Brasileiro_Uruguai_Oeste_Argentino_buffered", "R2", "darkblue",
        "Southeast Brazil, Uruguay and western Argentina",
    ),
    "Amazonia_norte_buffered": RegionMeta(
        "Amazonia_norte_buffered", "R3", "darkgreen", "Northern Amazon",
    ),
    "Amazonia_sul_buffered": RegionMeta(
        "Amazonia_sul_buffered", "R4", "lightgreen", "Southern Amazon",
    ),
    "Bacia_do_Prata_buffered": RegionMeta(
        "Bacia_do_Prata_buffered", "R5", "purple", "La Plata Basin",
    ),
    "Minas_Gerais_Sudeste_Brasileiro_buffered": RegionMeta(
        "Minas_Gerais_Sudeste_Brasileiro_buffered", "R6", "orange",
        "Minas Gerais and Southeast Brazil",
    ),
    "Regiao_Equatorial_buffered": RegionMeta(
        "Regiao_Equatorial_buffered", "R7", "saddlebrown", "Equatorial region",
    ),
    "Colombia_Maracaibo_buffered": RegionMeta(
        "Colombia_Maracaibo_buffered", "R8", "violet", "Colombia and Maracaibo",
    ),
}

DEFAULT_REGIONS: list[str] = list(REGION_META)

# Regions whose classification model was saved with the custom Cast layer.
# (In the original notebook this split the loading into two branches; here
# custom_objects is always passed, so the list is documentation only.)
REGIONS_WITH_CAST_LAYER = {
    "Amazonia_norte_buffered",
    "Amazonia_sul_buffered",
    "Regiao_Equatorial_buffered",
    "Nordeste_Brasileiro_buffered",
    "Minas_Gerais_Sudeste_Brasileiro_buffered",
    "Sudeste_Brasileiro_Uruguai_Oeste_Argentino_buffered",
}

# Model input variables, in the exact order used during training.
# Changing this order invalidates the predictions.
MODEL_VARIABLES: list[str] = [
    "cape", "msl", "div_250", "div_850",
    "kx", "totalx", "ur_600", "ur_700", "w",
    "ur_850", "dls", "lls", "d2m",
    "t2m", "grad_T",
]

# Maps the computed variable names onto the names the models expect.
RENAME_TO_MODEL = {
    "prmsl": "msl",
    "KI": "kx",
    "TT": "totalx",
    "div850": "div_850",
    "div250": "div_250",
    "ur850": "ur_850",
    "ur600": "ur_600",
    "td2m": "d2m",
    "ur700": "ur_700",
    "w500": "w",
}

VAR_PROB = "prob-flash-gfs"
VAR_DETER = "deter-flash-gfs"


# ============================================================
# Bounding box
# ============================================================


def read_shp_bounds(shp_path: Path) -> tuple[float, float, float, float]:
    """Read (minx, miny, maxx, maxy) from a .shp header.

    An ESRI header is 100 bytes long; the bounding box is four
    little-endian doubles starting at byte 36. This avoids a GDAL
    dependency.
    """
    with open(shp_path, "rb") as handle:
        header = handle.read(100)

    if len(header) < 100:
        raise ValueError(f"Truncated .shp header: {shp_path}")

    magic = struct.unpack(">i", header[0:4])[0]
    if magic != 9994:
        raise ValueError(f"Does not look like a valid shapefile: {shp_path}")

    minx, miny, maxx, maxy = struct.unpack("<4d", header[36:68])
    return minx, miny, maxx, maxy


def _load_bounds_file() -> dict[str, list[float]]:
    if BOUNDS_FILE.exists():
        with open(BOUNDS_FILE, encoding="utf-8") as handle:
            return json.load(handle)
    return {}


def get_region_bounds(
    region: str,
    shapefiles_root: Path | None = None,
) -> tuple[float, float, float, float]:
    """Resolve a region's bounding box.

    Order: config/region_bounds.json -> .shp header under shapefiles_root.
    """
    cached = _load_bounds_file().get(region)
    if cached and len(cached) == 4:
        return tuple(float(v) for v in cached)  # type: ignore[return-value]

    if shapefiles_root is not None:
        shp = shapefiles_root / f"{region}.shp"
        if shp.exists():
            return read_shp_bounds(shp)

    raise FileNotFoundError(
        f"No bounding box for '{region}'.\n"
        f"Provide it in {BOUNDS_FILE} or make "
        f"{region}.shp available under {shapefiles_root}.\n"
        f"Use scripts/extract_region_bounds.py to generate the JSON."
    )


def region_meta(region: str) -> RegionMeta:
    if region not in REGION_META:
        raise KeyError(f"Unknown region: {region}")
    return REGION_META[region]
