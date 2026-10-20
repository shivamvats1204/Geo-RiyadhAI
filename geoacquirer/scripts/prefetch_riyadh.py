"""Prefetch Overture layers for a Riyadh study area into the local cache (parallel, one connection per layer).

Usage: python scripts/prefetch_riyadh.py [west south east north]
Default bbox covers central/north Riyadh (Olaya, King Fahd Rd, Northern Ring Rd).
"""
import sys
import time
from concurrent.futures import ThreadPoolExecutor

from geoacquirer.providers.overture import OvertureProvider

DEFAULT_BBOX = (46.62, 24.64, 46.74, 24.78)


def fetch(entity, bbox):
    p = OvertureProvider()  # own DuckDB connection per thread
    t = time.time()
    b = p.fetch_features(entity, bbox)
    print(f"{entity}: {len(b.gdf)} rows in {time.time()-t:.0f}s (cache={b.provenance.from_cache}) release={b.provenance.dataset_version}", flush=True)
    return entity, len(b.gdf)


if __name__ == "__main__":
    bbox = tuple(float(x) for x in sys.argv[1:5]) if len(sys.argv) >= 5 else DEFAULT_BBOX
    print("bbox", bbox, flush=True)
    with ThreadPoolExecutor(4) as ex:
        for res in ex.map(lambda e: fetch(e, bbox), ["poi", "road", "connector", "building"]):
            pass
    print("done", flush=True)
