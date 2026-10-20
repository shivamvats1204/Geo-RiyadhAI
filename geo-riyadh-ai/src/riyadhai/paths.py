"""Where the prepared data lives. One place, so the app and every script agree.

Everything is under RIYADHAI_DATA_ROOT (default: your home folder):
  riyadh_city_store/     tiled GeoParquet built from the raw downloads (scripts/build_city_store.py)
    metro_network/       metro lines and stations (scripts/fetch_metro_network.py)
  riyadh_partitioned/    the same layers with district / neighbourhood assignments, the registry and the vector tiles
                         (scripts/build_partition.py, scripts/pregen_tiles.py)
  urbi_city/ suhail_city/                       raw downloads (geoacquirer/scripts/run_urbi_city.py, run_suhail_city.py)
  urbi_districts/ suhail_neighborhoods/ overture_districts/   administrative-area sources (scripts/prepare_*.py, fetch_*.py)
Individual locations can be overridden with RIYADHAI_CITYSTORE and RIYADHAI_PARTITION.
"""
from __future__ import annotations

import os
from pathlib import Path


def data_root() -> Path:
    return Path(os.environ.get("RIYADHAI_DATA_ROOT", Path.home()))


def city_store() -> Path:
    return Path(os.environ.get("RIYADHAI_CITYSTORE", data_root() / "riyadh_city_store"))


def partition_root() -> Path:
    return Path(os.environ.get("RIYADHAI_PARTITION", data_root() / "riyadh_partitioned"))


def raw_dir(name: str) -> Path:
    """Raw / intermediate folders: urbi_city, suhail_city, urbi_districts, suhail_neighborhoods, overture_districts."""
    return data_root() / name
