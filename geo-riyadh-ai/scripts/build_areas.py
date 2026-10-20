"""Prepare analysis areas (the data the analysis tools and the code agent work on) from the city store plus Overture open data.

An area is a bounding box. Overture layers (places, roads, buildings...) are fetched once per area (live, a few minutes) and cached; Urbi and Suhail layers
come from the city store. The app also does this on demand when you pick a district that no prepared area covers.

Usage: python scripts/build_areas.py [--data data_city] [--no-live] name=west,south,east,north [name=...]
  e.g. python scripts/build_areas.py north_riyadh=46.60,24.78,46.76,24.92
Default areas (if none given) are the four used by benchmarks/full_evaluation.py.
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from riyadhai.citystore import CityStore  # noqa: E402
from riyadhai.geoacquirer_client import GeoAcquirerClient  # noqa: E402
from riyadhai.paths import city_store  # noqa: E402
from riyadhai.store import UrbanStore, build_area_from_city  # noqa: E402

DEFAULTS = {"airport": (46.675, 24.935, 46.735, 24.985), "central_riyadh": (46.62, 24.64, 46.74, 24.78),
            "north_riyadh": (46.60, 24.78, 46.76, 24.92), "east_fringe": (47.10, 24.70, 47.22, 24.82)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("areas", nargs="*", help="name=west,south,east,north")
    ap.add_argument("--data", default="data_city", help="UrbanStore folder to write into")
    ap.add_argument("--no-live", action="store_true", help="only use already cached Overture layers")
    ns = ap.parse_args()
    areas = {a.split("=")[0]: tuple(float(v) for v in a.split("=")[1].split(",")) for a in ns.areas} or DEFAULTS
    city, store = CityStore(city_store()), UrbanStore(ns.data)
    for name, bbox in areas.items():
        t = time.time()
        try:
            ds = build_area_from_city(store, name, bbox, GeoAcquirerClient(), city, allow_live=not ns.no_live)
            print(name, "OK", round(time.time() - t), "s", {k: len(v) for k, v in ds.layers.items() if k in ("poi", "road_raw", "building", "parking", "urbi_building", "parcel")}, flush=True)
        except Exception as e:  # noqa: BLE001  report and continue with the next area
            print(name, "FAILED", type(e).__name__, str(e)[:300], flush=True)
    print("done")


if __name__ == "__main__":
    main()
