"""Generate config/region_bounds.json from the region shapefiles.

Run it once (in Colab with Drive mounted, or on a local machine) and commit
the resulting JSON. The server then needs neither the shapefiles nor
GDAL/geopandas.

    python scripts/extract_region_bounds.py /path/to/shapefiles_clusters_with_buffer
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.regions import DEFAULT_REGIONS, read_shp_bounds  # noqa: E402

OUTPUT = Path(__file__).resolve().parents[1] / "config" / "region_bounds.json"


def main() -> int:

    if len(sys.argv) < 2:
        print(__doc__)
        return 1

    root = Path(sys.argv[1])
    bounds: dict[str, list[float]] = {}

    for region in DEFAULT_REGIONS:
        shp = root / f"{region}.shp"
        if not shp.exists():
            print(f"[missing] {shp}")
            continue
        minx, miny, maxx, maxy = read_shp_bounds(shp)
        bounds[region] = [round(v, 4) for v in (minx, miny, maxx, maxy)]
        print(f"[ok] {region}: lon {minx:.2f} to {maxx:.2f}, lat {miny:.2f} to {maxy:.2f}")

    if not bounds:
        print("No shapefile found.")
        return 1

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(bounds, indent=2), encoding="utf-8")
    print(f"\nSaved to {OUTPUT} ({len(bounds)} regions)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
