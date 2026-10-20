"""Reproducible exports: CSV, GeoJSON, GeoParquet, JSONL, evidence manifest, report, map."""
from __future__ import annotations

import datetime as dt
import hashlib
import html
import json
from pathlib import Path
from typing import Optional

import geopandas as gpd

from .. import __version__

PASSED = {"PASS", "PASS_NEAR"}
FAILED = {"FAIL"}


def _split_constraints(evidence_json: str):
    ev = json.loads(evidence_json)
    passed, failed, unverified = [], [], []
    for e in ev:
        label = e["predicate"]
        if e["result"] == "PASS":
            passed.append(label)
        elif e["result"] == "PASS_NEAR":
            passed.append(f"{label} (within positional uncertainty)")
        elif e["result"] == "FAIL":
            failed.append(label)
        else:
            unverified.append(f"{label} [{e['result']}]")
    return passed, failed, unverified, ev


def flatten_results(res_metric: gpd.GeoDataFrame, query, run_info: dict) -> gpd.GeoDataFrame:
    """Result frame in WGS84 with human-readable evidence columns."""
    g = res_metric.to_crs(4326).copy()
    rows = []
    for _, r in g.iterrows():
        passed, failed, unverified, ev = _split_constraints(r["evidence_json"])
        nearest_j = next((e for e in ev if e["predicate"].startswith("AT_INTERSECTION")), None)
        centroid = r.geometry.centroid
        rows.append({
            "id": r.get("poi_id") or r.get("building_id") or r.get("parcel_id"),
            "name": r.get("name"), "category": r.get("category"),
            "lon": round(centroid.x, 7), "lat": round(centroid.y, 7),
            "status": r["status"],
            "constraints_passed": " | ".join(passed), "constraints_failed": " | ".join(failed),
            "constraints_unverified": " | ".join(unverified),
            "nearest_junction_id": nearest_j.get("object_id") if nearest_j else None,
            "nearest_junction_distance_m": nearest_j.get("nearest_distance_m") if nearest_j else None,
            "building_id": r.get("building_id_resolved"), "building_match": r.get("building_match"),
            "building_distance_m": r.get("building_distance_m"),
            "parcel_id": r.get("parcel_id_resolved"), "parcel_match": r.get("parcel_match"),
            "nearby_roads": r.get("nearby_roads"),
            "position_basis": "POI point (no mapped entrance); positional uncertainty %.0f m" % query.position_uncertainty_m
            if r.geometry.geom_type == "Point" else "polygon",
            "source": run_info.get("poi_source"), "source_record_dataset": r.get("source_dataset"),
            "source_release": run_info.get("release"), "retrieved_at": run_info.get("retrieved_at"),
            "confidence_source_reported": r.get("confidence"),
            # optional source attributes (parking: capacity is operator-reported, not live occupancy)
            "capacity_reported": r.get("capacity"), "access": r.get("access"), "is_paid": r.get("is_paid"),
            "schedule": r.get("schedule"), "address": r.get("address"), "for_trucks": r.get("for_trucks"),
            "evidence_json": r["evidence_json"], "geometry": r.geometry,
        })
    return gpd.GeoDataFrame(rows, geometry="geometry", crs=4326)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def export_all(flat: gpd.GeoDataFrame, query, run_dir: Path, formats: list[str], manifest: dict,
               map_layers: Optional[dict] = None, include_statuses: Optional[list[str]] = None) -> dict[str, Path]:
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    statuses = include_statuses or query.output.include_statuses
    exp = flat[flat["status"].isin(statuses)].copy()
    out: dict[str, Path] = {}
    if "csv" in formats:
        p = run_dir / "results.csv"
        exp.drop(columns="geometry").to_csv(p, index=False, encoding="utf-8-sig")
        out["csv"] = p
    if "geojson" in formats:
        p = run_dir / "results.geojson"
        exp.to_file(p, driver="GeoJSON")
        out["geojson"] = p
    if "geoparquet" in formats:
        p = run_dir / "results.parquet"
        exp.to_parquet(p)
        out["geoparquet"] = p
    if "jsonl" in formats:
        p = run_dir / "results.jsonl"
        with open(p, "w", encoding="utf-8") as f:
            for rec in exp.drop(columns="geometry").to_dict("records"):
                f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
        out["jsonl"] = p
    counts = flat["status"].value_counts().to_dict()
    manifest = dict(manifest)
    manifest.update({
        "geoacquirer_version": __version__,
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "natural_language_query": query.raw_text,
        "geoquery_ir": json.loads(query.model_dump_json()),
        "status_counts_all_candidates": counts,
        "exported_statuses": statuses,
        "n_exported": int(len(exp)),
        "schema": {c: str(t) for c, t in exp.drop(columns="geometry").dtypes.items()},
    })
    if "map" in formats:
        p = run_dir / "map.html"
        p.write_text(render_map(flat, query, map_layers or {}), encoding="utf-8")
        out["map"] = p
    p = run_dir / "report.html"
    p.write_text(render_report(flat, query, manifest), encoding="utf-8")
    out["report"] = p
    manifest["files"] = {k: {"path": v.name, "sha256": _sha256(v)} for k, v in out.items()}
    mp = run_dir / "manifest.json"
    mp.write_text(json.dumps(manifest, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    out["manifest"] = mp
    return out


_COLORS = {"VERIFIED_MATCH": "#1a9850", "PROBABLE_MATCH": "#fdae61", "INSUFFICIENT_EVIDENCE": "#999999", "REJECTED": "#d73027"}


def render_report(flat, query, manifest) -> str:
    counts = manifest.get("status_counts_all_candidates", {})
    prov = manifest.get("provenance", {})
    rows = "".join(f"<tr><td>{html.escape(k)}</td><td>{v}</td></tr>" for k, v in counts.items())
    defaults = "".join(f"<li><code>{html.escape(str(k))}</code> = {html.escape(str(v))}</li>" for k, v in query.defaults_applied.items())
    amb = "".join(f"<li>{html.escape(a)}</li>" for a in query.ambiguities)
    amb += "".join(f"<li><b>NOT APPLIED:</b> unrecognised phrase <code>{html.escape(a)}</code> did not filter the results</li>" for a in query.unparsed_phrases)
    prov_rows = "".join(
        f"<tr><td>{html.escape(str(k))}</td><td>{html.escape(str(v.get('dataset_version','')))}</td><td>{html.escape(str(v.get('license','')))}</td>"
        f"<td>{html.escape(str(v.get('retrieved_at','')))}</td><td>{'cache' if v.get('from_cache') else 'live'}</td></tr>"
        for k, v in prov.items())
    notes = "".join(f"<li>{html.escape(str(n))}</li>" for n in manifest.get("coverage_notes", []))
    ver = manifest.get("verification", {})
    log = "".join(f"<li>{html.escape(str(l))}</li>" for l in manifest.get("log", []))
    sample = flat[flat["status"].isin(["VERIFIED_MATCH", "PROBABLE_MATCH"])].head(25)
    srows = "".join(
        f"<tr><td>{html.escape(str(r['name']))}</td><td>{html.escape(str(r['category']))}</td><td>{r['lat']:.5f}, {r['lon']:.5f}</td>"
        f"<td>{r['status']}</td><td>{html.escape(str(r['constraints_passed']))}</td><td>{html.escape(str(r['building_match']))}</td></tr>"
        for _, r in sample.iterrows())
    return f"""<!doctype html><meta charset="utf-8"><title>GeoAcquirer report</title>
<style>body{{font:15px/1.5 system-ui,sans-serif;max-width:980px;margin:2rem auto;padding:0 1rem}}table{{border-collapse:collapse}}td,th{{border:1px solid #ccc;padding:4px 8px;text-align:left;vertical-align:top}}code{{background:#f3f3f3;padding:1px 4px}}</style>
<h1>GeoAcquirer analytical report</h1>
<p><b>Query:</b> {html.escape(query.raw_text)}</p>
<p><b>Area:</b> {html.escape(query.area.name)} &nbsp; <b>bbox:</b> {query.area.bbox}</p>
<h2>Outcome</h2><table><tr><th>Status</th><th>Candidates</th></tr>{rows}</table>
<p>Exported statuses: {', '.join(manifest.get('exported_statuses', []))} ({manifest.get('n_exported')} rows).
<i>VERIFIED</i> = every constraint passed with margin larger than the positional uncertainty. <i>PROBABLE</i> = passed, but within the uncertainty band
of at least one threshold. <i>INSUFFICIENT_EVIDENCE</i> = a constraint could not be verified or failed within the uncertainty band. <i>REJECTED</i> = a hard constraint failed.</p>
<h2>Defaults and ambiguities (all configurable)</h2><ul>{defaults}</ul><ul>{amb}</ul>
<h2>Sources, licences, acquisition dates</h2>
<table><tr><th>Layer</th><th>Release</th><th>Licence</th><th>Retrieved</th><th>Origin</th></tr>{prov_rows}</table>
<h2>Coverage and data-quality notes</h2><ul>{notes}</ul>
<h2>Independent verification</h2><pre>{html.escape(json.dumps(ver, indent=2))}</pre>
<h2>Sample of matches</h2><table><tr><th>Name</th><th>Category</th><th>Lat, Lon</th><th>Status</th><th>Constraints passed</th><th>Building link</th></tr>{srows}</table>
<h2>Processing log</h2><ul>{log}</ul>
<h2>Reproduce</h2><pre>{html.escape(manifest.get('reproduce', ''))}</pre>
"""


def _gj(gdf, cols, max_rows=None):
    g = gdf if max_rows is None else gdf.head(max_rows)
    return json.loads(g[cols + ["geometry"]].to_json()) if len(g) else {"type": "FeatureCollection", "features": []}


def render_map(flat: gpd.GeoDataFrame, query, layers: dict) -> str:
    pts = flat[flat["status"].isin(["VERIFIED_MATCH", "PROBABLE_MATCH", "INSUFFICIENT_EVIDENCE"])]
    pts_gj = _gj(pts, ["name", "category", "status", "constraints_passed", "constraints_unverified", "building_match", "nearby_roads"])
    hw = layers.get("highways")
    jn = layers.get("junctions")
    hw_gj = _gj(hw, ["road_name", "class"]) if hw is not None and len(hw) else {"type": "FeatureCollection", "features": []}
    jn_gj = _gj(jn, ["kind", "road_names"]) if jn is not None and len(jn) else {"type": "FeatureCollection", "features": []}
    minx, miny, maxx, maxy = (flat.total_bounds if len(flat) else (46.6, 24.6, 46.8, 24.8))
    return f"""<!doctype html><meta charset="utf-8"><title>GeoAcquirer map</title>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css">
<style>html,body,#m{{height:100%;margin:0}}.lg{{background:#fff;padding:8px 10px;line-height:1.5;font:13px system-ui;border-radius:4px;box-shadow:0 0 6px #0004}}.lg i{{display:inline-block;width:11px;height:11px;border-radius:50%;margin-right:6px}}</style>
<div id="m"></div>
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<script>
const C={json.dumps(_COLORS)};
const map=L.map('m').fitBounds([[{miny},{minx}],[{maxy},{maxx}]]);
L.tileLayer('https://tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png',{{maxZoom:19,attribution:'© OpenStreetMap contributors; data © Overture Maps Foundation'}}).addTo(map);
const hw=L.geoJSON({json.dumps(hw_gj)},{{style:{{color:'#7b3294',weight:3,opacity:.6}},onEachFeature:(f,l)=>l.bindPopup((f.properties.road_name||'(unnamed)')+' ['+f.properties.class+']')}}).addTo(map);
const jn=L.geoJSON({json.dumps(jn_gj)},{{pointToLayer:(f,ll)=>L.circleMarker(ll,{{radius:5,color:'#2166ac',fillColor:'#92c5de',fillOpacity:.9,weight:1}}),onEachFeature:(f,l)=>l.bindPopup('four-way junction<br>'+(f.properties.road_names||''))}}).addTo(map);
const res=L.geoJSON({json.dumps(pts_gj)},{{pointToLayer:(f,ll)=>L.circleMarker(ll,{{radius:6,color:'#222',weight:1,fillColor:C[f.properties.status],fillOpacity:.9}}),
 onEachFeature:(f,l)=>{{const p=f.properties;l.bindPopup('<b>'+(p.name||'(unnamed)')+'</b><br>'+p.category+'<br><b>'+p.status+'</b><br>passed: '+(p.constraints_passed||'-')+'<br>unverified: '+(p.constraints_unverified||'-')+'<br>building: '+(p.building_match||'-')+'<br>roads: '+(p.nearby_roads||'-'));}}}}).addTo(map);
L.control.layers(null,{{'Results':res,'Four-way junctions':jn,'Motorway/trunk':hw}}).addTo(map);
const lg=L.control({{position:'bottomright'}});lg.onAdd=()=>{{const d=L.DomUtil.create('div','lg');d.innerHTML='<b>Status</b><br>'+Object.entries(C).map(([k,v])=>'<i style="background:'+v+'"></i>'+k).join('<br>')+'<br><small>Query: {html.escape(query.raw_text)[:120]}</small>';return d}};lg.addTo(map);
</script>"""
