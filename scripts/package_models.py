"""Pack the trained models into a single .zip to upload to Drive.

Run it in Colab with Drive mounted:

    python scripts/package_models.py \
        --classification /content/drive/MyDrive/Doutorado/Model-Trained \
        --regressor      /content/drive/MyDrive/Doutorado/Model-Trained-new \
        --shapefiles     /content/drive/MyDrive/Doutorado/shapefiles_clusters_with_buffer \
        --out            /content/drive/MyDrive/Doutorado/lightning_assets.zip

Then take the file id from Drive and set GDRIVE_ASSETS_ZIP_ID on Render.
The zip carries only the files the API uses, not the whole folder.
"""

from __future__ import annotations

import argparse
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.regions import DEFAULT_REGIONS  # noqa: E402

STEPS = "6H"


def main() -> int:

    parser = argparse.ArgumentParser(description="Pack the models into a zip")
    parser.add_argument("--classification", required=True)
    parser.add_argument("--regressor", required=True)
    parser.add_argument("--shapefiles")
    parser.add_argument("--out", required=True)
    parser.add_argument("--steps", default=STEPS)
    args = parser.parse_args()

    classification = Path(args.classification)
    regressor = Path(args.regressor)
    out = Path(args.out)

    included = 0
    missing: list[str] = []

    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as archive:

        for region in DEFAULT_REGIONS:

            steps = args.steps

            class_dir = classification / region / "results" / "train_data" / "classification"
            reg_dir = regressor / region / "results" / "train_data" / "regressor"

            items = [
                (class_dir / f"nw_classification_model_{region}_{steps}.h5",
                 f"Model-Trained/{region}/results/train_data/classification"),
                (class_dir / f"nw_mean_{region}_{steps}.npy",
                 f"Model-Trained/{region}/results/train_data/classification"),
                (class_dir / f"nw_std_{region}_{steps}.npy",
                 f"Model-Trained/{region}/results/train_data/classification"),
                (reg_dir / f"regressor_model_{region}_{steps}.keras",
                 f"Model-Trained-new/{region}/results/train_data/regressor"),
                (reg_dir / f"mean_regressor_{region}_{steps}.npy",
                 f"Model-Trained-new/{region}/results/train_data/regressor"),
                (reg_dir / f"std_regressor_{region}_{steps}.npy",
                 f"Model-Trained-new/{region}/results/train_data/regressor"),
                (reg_dir / f"mean_data_regressor_{region}_{steps}.npy",
                 f"Model-Trained-new/{region}/results/train_data/regressor"),
                (reg_dir / f"std_data_regressor_{region}_{steps}.npy",
                 f"Model-Trained-new/{region}/results/train_data/regressor"),
            ]

            for source, arc_dir in items:
                if source.exists():
                    archive.write(source, f"{arc_dir}/{source.name}")
                    included += 1
                else:
                    missing.append(str(source))

            if args.shapefiles:
                shp_root = Path(args.shapefiles)
                for ext in (".shp", ".shx", ".dbf", ".prj", ".cpg"):
                    source = shp_root / f"{region}{ext}"
                    if source.exists():
                        archive.write(
                            source,
                            f"shapefiles_clusters_with_buffer/{source.name}",
                        )
                        included += 1

    print(f"\n{out} -> {included} files, {out.stat().st_size / 1e6:.1f} MB")

    if missing:
        print(f"\nNot found ({len(missing)}):")
        for item in missing:
            print(f"  {item}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
