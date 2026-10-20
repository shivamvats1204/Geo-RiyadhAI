"""Canonical urban data store.

Logical zones (directories under `root`): raw/ (pointer to GeoAcquirer cache + provenance), canonical/<area>/ (deduplicated
GeoParquet layers, WGS84), derived/<area>/ (junctions, features), models/, analyses/ (queries, plans, results, scenarios,
reports) and provenance/ (per-layer source, licence, release, retrieval date, transformations).
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import box

from geoacquirer.engine import DataContext
from geoacquirer.ontology import category_mask
from geoacquirer.providers.overture import connectors_frame, segments_frame
from geoacquirer.spatial import to_metric, utm_crs_for_bbox

from .provenance import data_date
from .geoacquirer_client import AcquisitionResult, GeoAcquirerClient

ZONES = ["raw", "canonical", "derived", "models", "analyses", "provenance"]


@dataclass
class AreaDataset:
    name: str
    bbox: tuple
    crs: object
    layers: dict[str, gpd.GeoDataFrame]
    connectors: Optional[gpd.GeoDataFrame]
    provenance: dict[str, dict]
    notes: list[str] = field(default_factory=list)
    _ctx: Optional[DataContext] = None
    scope: Optional[dict] = None  # {'name', 'uid', 'polygon' (metric CRS)}: the district / neighbourhood the user selected; every analysis is limited to it
    _network = None

    # ------------------------------------------------------------ views
    def has(self, name: str) -> bool:
        g = self.layers.get(name)
        return g is not None and len(g) > 0

    def get(self, name: str) -> gpd.GeoDataFrame:
        if name not in self.layers:
            raise KeyError(f"layer {name!r} not available for area {self.name!r}; available: {sorted(self.layers)}")
        return self.layers[name]

    def full_polygon(self):
        """The whole prepared data extent (a box), ignoring the selected scope."""
        return to_metric(gpd.GeoDataFrame(geometry=[box(*self.bbox)], crs=4326), self.crs).geometry.iloc[0]

    def area_polygon(self):
        """The analysis region: the selected district / neighbourhood clipped to the prepared data, or the whole prepared extent when none is selected."""
        p = self.full_polygon()
        return p.intersection(self.scope["polygon"]) if self.scope else p

    def scope_label(self) -> str:
        return self.scope["name"] if self.scope else "the prepared data extent"

    def network(self):
        if self._network is None:
            from geoacquirer.spatial.network import RoadNetwork
            self._network = RoadNetwork(self.layers["road"], self.connectors)
        return self._network

    def geoacquirer_context(self) -> DataContext:
        """DataContext so GeoAcquirer's deterministic evaluator can run constraints on this area."""
        if self._ctx is None:
            ctx = DataContext(crs=self.crs)
            for k in ("poi", "building", "parcel", "district"):
                if k in self.layers:
                    ctx.layers[k] = self.layers[k]
            if "parking" in self.layers:
                ctx.layers["parking"] = self.layers["parking"]
            ctx.layers["road"] = self.layers["road"]
            ctx.layers["junction"] = self.layers["junction"]
            ctx.connectors = self.connectors
            ctx.area = self.area_polygon()
            ctx.provenance = dict(self.provenance)
            self._ctx = ctx
        return self._ctx

    def coverage(self) -> dict:
        out = {"name": self.name, "bbox": list(self.bbox), "layers": {}}
        for k, g in self.layers.items():
            out["layers"][k] = {"rows": int(len(g)), "geometry": sorted(g.geometry.geom_type.unique().tolist()) if len(g) else []}
        j = self.layers.get("junction")
        if j is not None:
            out["junction_kinds"] = j["kind"].value_counts().to_dict()
        net = self.layers.get("road")
        if net is not None and self.connectors is not None:
            sizes = self.network().component_sizes()
            out["road_network"] = {"components": len(sizes), "largest_component_nodes": sizes[0] if sizes else 0,
                                   "share_in_largest": round(sizes[0] / max(1, sum(sizes)), 4) if sizes else 0}
        return out


class UrbanStore:
    def __init__(self, root: Path | str = "data"):
        self.root = Path(root)
        for z in ZONES:
            (self.root / z).mkdir(parents=True, exist_ok=True)
        self._areas: dict[str, AreaDataset] = {}

    # --------------------------------------------------------------- build
    def build_area(self, name: str, bbox: tuple, client: GeoAcquirerClient, allow_live: bool = True) -> AreaDataset:
        acq = client.fetch_layers(bbox, allow_live=allow_live)
        return self._assemble(name, bbox, acq)

    def _assemble(self, name: str, bbox: tuple, acq: AcquisitionResult) -> AreaDataset:
        crs = utm_crs_for_bbox(bbox)
        L = {k: to_metric(v, crs) for k, v in acq.layers.items() if v is not None}
        notes = list(acq.warnings)
        poi = L["poi"]
        if "category_path" in poi:
            parking = poi[category_mask(poi["category_path"], ["parking"])].copy()
            parking["source"] = "overture_places"
            parking["capacity"] = np.nan
            parking = parking.reset_index(drop=True)
            if not isinstance(parking, gpd.GeoDataFrame):  # a sparse area can have no POI at all; keep the layer spatially typed
                parking = gpd.GeoDataFrame(parking, geometry="geometry", crs=crs) if "geometry" in parking else gpd.GeoDataFrame(parking, geometry=[], crs=crs)
            L["parking"] = parking
        ctx = DataContext(crs=crs)
        L["road_raw"] = L["road"]
        seg = segments_frame(L["road"])
        con = connectors_frame(L["connector"])
        ctx.add_roads(seg, con)
        L["road"] = ctx.layers["road"]
        L["junction"] = ctx.layers["junction"]
        L["excluded_crossings"] = ctx.layers["excluded_crossings"]
        notes += ctx.notes
        ds = AreaDataset(name, tuple(bbox), crs, L, con, acq.provenance, notes)
        self._areas[name] = ds
        self._save(ds)
        return ds

    # ----------------------------------------------------------- persistence
    def _save(self, ds: AreaDataset) -> None:
        d = self.root / "canonical" / ds.name
        d.mkdir(parents=True, exist_ok=True)
        for k, g in ds.layers.items():
            if k in ("junction", "excluded_crossings", "road", "connector"):  # derived or list-typed; rebuilt on load from road_raw
                continue
            g.to_crs(4326).to_parquet(d / f"{k}.parquet")
        ds.connectors.rename(columns={"id": "connector_id"}).to_crs(4326).to_parquet(d / "connector.parquet")
        (d / "meta.json").write_text(json.dumps({"name": ds.name, "bbox": list(ds.bbox), "saved_at": time.time(), "notes": ds.notes}, indent=2))
        (self.root / "provenance" / f"{ds.name}.json").write_text(json.dumps(ds.provenance, indent=2, default=str))

    def load_area(self, name: str) -> AreaDataset:
        if name in self._areas:
            return self._areas[name]
        d = self.root / "canonical" / name
        if not (d / "meta.json").exists():
            raise FileNotFoundError(f"area {name!r} not in store; build it first")
        meta = json.loads((d / "meta.json").read_text())
        bbox = tuple(meta["bbox"])
        crs = utm_crs_for_bbox(bbox)
        layers = {p.stem: to_metric(gpd.read_parquet(p), crs) for p in d.glob("*.parquet")}
        for g in layers.values():  # reprojection can invalidate thin repaired polygons: repair once at load so every analysis sees valid geometry
            if len(g) and g.geometry.geom_type.isin(["Polygon", "MultiPolygon"]).any():
                bad = ~g.geometry.is_valid
                if bad.any():
                    g.loc[bad, "geometry"] = g.geometry[bad].make_valid()
        con = connectors_frame(layers.pop("connector"))
        ctx = DataContext(crs=crs)
        # rebuild topology from the canonical raw road layer (deterministic)
        ctx.add_roads(segments_frame(layers["road_raw"]), con)
        layers["road"], layers["junction"], layers["excluded_crossings"] = ctx.layers["road"], ctx.layers["junction"], ctx.layers["excluded_crossings"]
        prov_path = self.root / "provenance" / f"{name}.json"
        prov = json.loads(prov_path.read_text()) if prov_path.exists() else {}
        ds = AreaDataset(name, bbox, crs, layers, con, prov, meta.get("notes", []) + ctx.notes)
        self._areas[name] = ds
        return ds

    def areas(self) -> list[str]:
        return sorted(p.parent.name for p in (self.root / "canonical").glob("*/meta.json"))

    # --------------------------------------------------------- user-supplied data
    def ingest_local(self, ds: AreaDataset, path: Path | str, entity_type: str, licence: str = "user-supplied") -> int:
        """Add a user-supplied parking or parcel file (CSV with lon/lat, GeoJSON, GeoParquet) to an area."""
        from geoacquirer.providers.local import _read
        g = _read(Path(path)).to_crs(ds.crs)
        g = g[g.geometry.intersects(ds.area_polygon())].reset_index(drop=True)
        if entity_type == "parking":
            if "capacity" not in g:
                g["capacity"] = np.nan
            g["source"] = f"local:{Path(path).name}"
            if "parking_id" not in g:
                g["parking_id"] = [f"local_{i}" for i in range(len(g))]
            g["poi_id"] = g["parking_id"].astype(str)
            if "category_path" not in g:
                g["category_path"] = [["travel_and_transportation", "parking"]] * len(g)
            ds.layers["parking"] = pd.concat([ds.layers.get("parking", g.iloc[:0]), g], ignore_index=True)
            ds.layers["parking"] = gpd.GeoDataFrame(ds.layers["parking"], geometry="geometry", crs=ds.crs)
        elif entity_type == "parcel":
            if "parcel_id" not in g:
                g["parcel_id"] = [f"local_{i}" for i in range(len(g))]
            ds.layers["parcel"] = g
        else:
            raise ValueError("entity_type must be 'parking' or 'parcel'")
        ds.provenance[entity_type] = {"provider_id": "local_files", "dataset_version": Path(path).name, "license": licence,
                                      "retrieved_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "from_cache": True}
        ds._ctx = None
        self._save(ds)
        return len(g)


def attach_city_layers(store: "UrbanStore", ds: AreaDataset, city) -> dict:
    """Add the Urbi/Suhail layers for ds.bbox from a CityStore (tiled GeoParquet) to an area, then persist it."""
    bbox = ds.bbox
    rep = {}
    for src, tgt in [("urbi_building", "urbi_building"), ("street", "street"), ("attraction", "attraction"), ("metro", "metro"), ("parcel", "parcel")]:
        g = city.query(src, bbox)
        if g.empty:
            rep[tgt] = 0
            continue
        gm = g.to_crs(ds.crs).reset_index(drop=True)
        bad = ~gm.geometry.is_valid
        if bad.any():  # reprojection can invalidate very thin repaired polygons
            gm.loc[bad, "geometry"] = gm.geometry[bad].make_valid()
        ds.layers[tgt] = gm
        ds.provenance[tgt] = {"provider_id": "suhail" if tgt == "parcel" else "urbi", "dataset_version": f"city store {data_date()}", "retrieved_at": data_date(),
                              "license": "user-stated permission (no document seen)", "from_cache": True}
        rep[tgt] = int(len(g))
    up = city.query("urbi_parking", bbox)
    if not up.empty:
        up = up.to_crs(ds.crs).reset_index(drop=True)
        addr = up["address"].where(up["address"].astype(str).str.len() > 0)
        pk = gpd.GeoDataFrame({"poi_id": up["urbi_id"].astype(str), "name": [a if isinstance(a, str) else f"Parking {i}" for a, i in zip(addr, up["urbi_id"])],
                               "category": up["subtype"].fillna("parking"), "category_path": [["travel_and_transportation", "parking"]] * len(up),
                               "capacity": pd.to_numeric(up["capacity"], errors="coerce"), "access": up["access"], "is_paid": up["is_paid"], "source": "urbi"},
                              geometry=up.geometry.values, crs=ds.crs)
        old = ds.layers.get("parking")
        ds.layers["parking"] = gpd.GeoDataFrame(pd.concat([old[old["source"] == "overture_places"] if old is not None and "source" in old else old, pk], ignore_index=True),
                                                geometry="geometry", crs=ds.crs)
        ds.provenance["parking"] = {"provider_id": "urbi+overture", "dataset_version": f"city store {data_date()}", "retrieved_at": data_date(),
                                    "license": "mixed (see provenance of each source)", "from_cache": True}
        rep["parking"] = int(len(ds.layers["parking"]))
    ds._ctx = None
    store._save(ds)
    return rep


def build_area_from_city(store: "UrbanStore", name: str, bbox: tuple, client, city, allow_live: bool = True) -> AreaDataset:
    ds = store.build_area(name, bbox, client, allow_live=allow_live)
    ds.notes.append("city layers attached: " + json.dumps(attach_city_layers(store, ds, city)))
    return ds
