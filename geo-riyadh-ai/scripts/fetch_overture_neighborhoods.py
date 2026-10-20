"""Overture Maps neighbourhood-type areas (macrohood / neighbourhood / microhood) for the whole city, from the open Overture release.

Output: <data root>/overture_districts/overture_neighborhoods.parquet. Open data (ODbL / CDLA, attribution required, see NOTICE.md); no authorization needed.
A scan of the release over the bbox takes a minute or two; results are cached by GeoAcquirer.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from geoacquirer.providers.overture import OvertureProvider  # noqa: E402
from riyadhai.paths import raw_dir  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bbox", type=float, nargs=4, default=(46.2, 24.2, 47.5, 25.4), metavar=("W", "S", "E", "N"))
    ns = ap.parse_args()
    batch = OvertureProvider().fetch_features("district", tuple(ns.bbox), max_age_days=None)
    gdf = batch.gdf if hasattr(batch, "gdf") else batch[0].gdf
    out = raw_dir("overture_districts")
    out.mkdir(parents=True, exist_ok=True)
    gdf.to_parquet(out / "overture_neighborhoods.parquet")
    print(len(gdf), "areas", gdf.subtype.value_counts().to_dict(), "->", out / "overture_neighborhoods.parquet")


if __name__ == "__main__":
    main()
