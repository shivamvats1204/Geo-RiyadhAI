"""Assign every feature of the city store to a district and neighbourhood, so the whole city can be browsed and counted by district.

For each tiled layer (buildings, parcels, parking, streets, attractions, metro) every feature gets
  urbi_district      uid of the Urbi district polygon containing its representative point (smallest polygon wins when they overlap)
  urbi_method        'within' | 'nearest' (outside every polygon: nearest district; `urbi_dist_m` says how far)
  suhail_nbh         uid of the Suhail neighbourhood polygon containing it ('nearest' fallback likewise, `suhail_dist_m`)
  overture_nbh       same for the Overture neighbourhood polygons
No feature is dropped: the partitioned store has exactly the row count of the source store (checked and reported).
Urbi records also carry their own district name; `urbi_name_agrees` records whether that matches the spatially assigned district.
"""
from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely

METRES_PER_DEG = 105_000.0  # rough degrees->metres at Riyadh's latitude, only used for the 'nearest' cutoff and its reported distance
NEAREST_MAX_M = 300.0


class Assigner:
    def __init__(self, polys: gpd.GeoDataFrame):
        p = polys[polys.has_polygon].reset_index(drop=True)
        self.uids = p.uid.to_numpy()
        self.names = p.name.fillna(p.name_ar).to_numpy()
        self.geoms = p.geometry.to_numpy()
        self.area = shapely.area(self.geoms)
        shapely.prepare(self.geoms)
        self.tree = shapely.STRtree(self.geoms)

    def assign(self, pts: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        n = len(pts)
        idx = np.full(n, -1)
        px, py = shapely.get_x(pts), shapely.get_y(pts)
        pi, gi = self.tree.query(pts)  # bbox candidates, then one vectorised exact test per polygon (polygons are prepared once)
        hit_p, hit_g = [], []
        order = np.argsort(gi, kind="stable")
        pi, gi = pi[order], gi[order]
        starts = np.flatnonzero(np.r_[True, gi[1:] != gi[:-1]]) if len(gi) else []
        for a, b in zip(starts, list(starts[1:]) + [len(gi)]):
            g = gi[a]
            ps = pi[a:b]
            ok = shapely.contains_xy(self.geoms[g], px[ps], py[ps])
            hit_p.append(ps[ok]); hit_g.append(np.full(int(ok.sum()), g))
        if hit_p:
            pi, gi = np.concatenate(hit_p), np.concatenate(hit_g)
            if len(pi):
                df = pd.DataFrame({"p": pi, "g": gi, "a": self.area[gi]}).sort_values(["p", "a"]).drop_duplicates("p")
                idx[df.p.to_numpy()] = df.g.to_numpy()
        method = np.where(idx >= 0, "within", "none").astype(object)
        dist = np.zeros(n)
        miss = np.where(idx < 0)[0]
        if len(miss):
            qi, gj = self.tree.query_nearest(pts[miss], all_matches=False)
            d = shapely.distance(pts[miss[qi]], self.geoms[gj]) * METRES_PER_DEG
            near = d <= NEAREST_MAX_M  # a feature just outside a boundary (digitising slack) is attached; one far away stays unassigned
            idx[miss[qi][near]] = gj[near]
            method[miss[qi][near]] = "nearest"
            dist[miss[qi]] = d
        return idx, method, dist

    def uid_of(self, idx):
        return np.where(idx >= 0, self.uids[np.maximum(idx, 0)], None)

    def name_of(self, idx):
        return np.where(idx >= 0, self.names[np.maximum(idx, 0)], None)


def build(src_root: Path, dst_root: Path, registry: gpd.GeoDataFrame, layers: list[str], log=print) -> dict:
    reg = registry
    urbi = Assigner(reg[(reg.source == "urbi") & (reg.kind == "district")])
    suh = Assigner(reg[reg.source == "suhail"])
    ove = Assigner(reg[reg.source == "overture"])
    report: dict = {}
    counts: list[dict] = []
    for layer in layers:
        sd, dd = Path(src_root) / layer, Path(dst_root) / layer
        if dd.exists():
            shutil.rmtree(dd)
        dd.mkdir(parents=True)
        t0, n_in, n_out = time.time(), 0, 0
        stats = {"urbi_nearest": 0, "suhail_nearest": 0, "overture_nearest": 0, "urbi_none": 0, "suhail_none": 0, "overture_none": 0, "name_checked": 0, "name_agrees": 0, "points": 0, "polygons": 0, "lines": 0}
        for f in sorted(sd.glob("t*__*.parquet")):
            g = gpd.read_parquet(f)
            n_in += len(g)
            pts = g.geometry.representative_point().to_numpy()
            ui, um, ud = urbi.assign(pts)
            si, sm, sdist = suh.assign(pts)
            oi, om, odist = ove.assign(pts)
            g["urbi_district"], g["urbi_district_name"], g["urbi_method"], g["urbi_dist_m"] = urbi.uid_of(ui), urbi.name_of(ui), um, np.round(ud, 1)
            g["suhail_nbh"], g["suhail_nbh_name"], g["suhail_method"], g["suhail_dist_m"] = suh.uid_of(si), suh.name_of(si), sm, np.round(sdist, 1)
            g["overture_nbh"], g["overture_nbh_name"], g["overture_method"] = ove.uid_of(oi), ove.name_of(oi), om
            if "district" in g.columns and layer != "parcel":
                rec = g["district"].astype("string").str.strip().str.lower().str.replace(r"\s+district$", "", regex=True)
                ass = pd.Series(g["urbi_district_name"], index=g.index).astype("string").str.strip().str.lower()
                ok = rec.notna() & (rec != "")
                g["urbi_name_agrees"] = np.where(ok, (rec == ass).to_numpy(), None)
                stats["name_checked"] += int(ok.sum())
                stats["name_agrees"] += int((ok & (rec == ass)).sum())
            gt = g.geometry.geom_type
            stats["points"] += int(gt.isin(["Point", "MultiPoint"]).sum())
            stats["polygons"] += int(gt.isin(["Polygon", "MultiPolygon"]).sum())
            stats["lines"] += int(gt.isin(["LineString", "MultiLineString"]).sum())
            stats["urbi_nearest"] += int((um == "nearest").sum())
            stats["suhail_nearest"] += int((sm == "nearest").sum())
            stats["overture_nearest"] += int((om == "nearest").sum())
            stats["urbi_none"] += int((um == "none").sum()); stats["suhail_none"] += int((sm == "none").sum()); stats["overture_none"] += int((om == "none").sum())
            g.to_parquet(dd / f.name)
            n_out += len(g)
            counts.append(pd.DataFrame({"layer": layer, "urbi_district": g["urbi_district"], "suhail_nbh": g["suhail_nbh"], "overture_nbh": g["overture_nbh"]}))
        for extra in ("index.json", "extent.json"):
            if (sd / extra).exists():
                shutil.copy(sd / extra, dd / extra)
        report[layer] = {"rows_in": n_in, "rows_out": n_out, "lossless": n_in == n_out, **stats, "seconds": round(time.time() - t0, 1)}
        log(layer, report[layer])
    allc = pd.concat(counts, ignore_index=True)
    allc.to_parquet(Path(dst_root) / "assignments_flat.parquet")
    (Path(dst_root) / "partition_report.json").write_text(json.dumps(report, indent=1))
    return report
