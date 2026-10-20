"""Raw downloads -> the tiled city store (about 20-40 minutes for the whole of Riyadh).

Steps
  1. convert the Urbi JSONL dumps (buildings, parking, streets, attractions, metro) to GeoParquet parts, then check and repair them
  2. tile them into 0.05 degree GeoParquet files (`CityStore`) with an index
  3. tile the Suhail parcels (with the first zoning rule per parcel) the same way
Inputs (default locations; see riyadhai/paths.py): <data root>/urbi_city/*_data.jsonl and <data root>/suhail_city/{parcels.parquet,rules.csv}
Output: <data root>/riyadh_city_store/

Usage: python scripts/build_city_store.py [--skip-urbi] [--skip-suhail]
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from riyadhai import citystore, urbi_city, urbi_quality  # noqa: E402
from riyadhai.paths import city_store, raw_dir  # noqa: E402

URBI_LAYERS = ["attractions", "buildings", "metro", "parking", "streets"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--urbi-dir", type=Path, default=raw_dir("urbi_city"))
    ap.add_argument("--suhail-dir", type=Path, default=raw_dir("suhail_city"))
    ap.add_argument("--skip-urbi", action="store_true")
    ap.add_argument("--skip-suhail", action="store_true")
    ns = ap.parse_args()
    city = citystore.CityStore(city_store())
    report = {}
    if not ns.skip_urbi:
        out = ns.urbi_dir / "parquet"
        out.mkdir(parents=True, exist_ok=True)
        for layer in URBI_LAYERS:
            t = time.time()
            conv = urbi_city.convert_layer(ns.urbi_dir, layer, out)
            qual = urbi_quality.check_and_repair(out, layer)
            report[f"urbi_{layer}"] = {"convert": conv, "quality": qual, "seconds": round(time.time() - t)}
            print("urbi", layer, report[f"urbi_{layer}"]["seconds"], "s", flush=True)
        report["city_store_urbi"] = citystore.build_from_urbi(city, out)
        print("tiled Urbi layers:", report["city_store_urbi"], flush=True)
    if not ns.skip_suhail:
        report["city_store_suhail"] = citystore.build_from_suhail(city, ns.suhail_dir)
        print("tiled Suhail parcels:", report["city_store_suhail"], flush=True)
    (city.root / "build_report.json").write_text(json.dumps(report, indent=1, default=str), encoding="utf-8")
    print("done:", city.summary())


if __name__ == "__main__":
    main()
