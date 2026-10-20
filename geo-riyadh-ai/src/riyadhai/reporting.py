"""Evidence report generation. Statements are grouped as OBSERVED (measured from data), MODEL OUTPUT (computed by a model/solver)
and INFERRED (a recommendation derived from those), and every one carries the id of the calculation that produced it."""
from __future__ import annotations

import html
import json
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import __version__

SECTION = {"observed": "Observed facts (measured from the data)", "model": "Model and solver outputs", "inferred": "Inferred recommendations"}
REFERENCE = "reference"  # retrieved methodology passages are shown in their own section, not as findings


def _table(df: pd.DataFrame, max_rows: int = 30) -> str:
    if df is None or len(df) == 0:
        return "<p><i>no rows</i></p>"
    d = df.head(max_rows).copy()
    for c in d.columns:
        if d[c].dtype.kind == "f":
            d[c] = d[c].round(3)
    return d.to_html(index=False, border=0, classes="t", escape=True)


def _layers(job, ctx) -> list[dict]:
    layers = []
    palette = {"rank_site_suitability": "#d73027", "optimize_facility_locations": "#1a9850", "run_geoquery": "#4575b4", "filter_features": "#4575b4",
               "aggregate_spatial_statistics": "#999999", "find_intersections": "#762a83"}
    for sid, r in job.results.items():
        if not r.result_id or r.tool not in palette:
            continue
        g = ctx.results.get(r.result_id).gdf
        cols = [c for c in g.columns if c != "geometry" and g[c].dtype.kind in "OifUb" and not g[c].map(lambda v: isinstance(v, (list, dict, np.ndarray))).any()][:8]
        g = g.head(1500)
        if r.tool == "aggregate_spatial_statistics":
            g = g[g["count"] > 0] if "count" in g else g
            cols = [c for c in ("name", "count", "density_per_km2") if c in g.columns]
        layers.append({"id": sid, "tool": r.tool, "color": palette[r.tool], "geojson": json.loads(g[cols + ["geometry"]].to_json())})
    return layers


def build_report(job, ctx) -> Path:
    out = Path(ctx.out_dir) / f"report_{job.id}.html"
    prov = ctx.area.provenance
    prov_rows = "".join(
        f"<tr><td>{html.escape(k)}</td><td>{html.escape(str(v.get('provider_id', '')))}</td><td>{html.escape(str(v.get('dataset_version', '')))}</td>"
        f"<td>{html.escape(str(v.get('license', '')))}</td><td>{html.escape(str(v.get('retrieved_at', '')))}</td></tr>" for k, v in prov.items())
    by = {k: [f for f in job.facts if f["type"] == k] for k in SECTION}
    facts_html = "".join(
        f"<h3>{SECTION[k]}</h3><ul>" + "".join(f"<li>{html.escape(f['text'])} <code>{f['calc']}</code></li>" for f in v) + "</ul>" for k, v in by.items() if v)
    tables = ""
    for sid, r in job.results.items():
        if r.table is not None and r.tool != "retrieve_reference_documents":
            tables += f"<h3>{html.escape(r.tool)} <code>{r.calc_id}</code></h3>{_table(r.table)}"
    rag = "".join(f"<blockquote><b>{html.escape(h['cite'])}</b> {html.escape(h['text'][:600])}</blockquote>"
                  for r in job.results.values() if r.tool == "retrieve_reference_documents" for h in r.summary.get("hits", []))
    suit = next((r for r in job.results.values() if r.tool == "rank_site_suitability"), None)
    expl = ""
    if suit is not None:
        top = ctx.results.get(suit.result_id).gdf
        expl = "<h3>Why each location ranks where it does</h3><ol>" + "".join(
            f"<li><b>{html.escape(str(r.cell_id))}</b> (score {r.score:.3f}, in the top-{len(top)} in {r.top_k_frequency:.0%} of sensitivity runs)<br><small>{html.escape(str(r.explanation))}</small></li>"
            for r in top.itertuples()) + "</ol>"
        sens = suit.summary["sensitivity"]
        expl += f"<pre>{html.escape(json.dumps({'weights': suit.summary['weights'], 'sensitivity': sens, 'filters': suit.summary['filters']}, indent=2, default=str))}</pre>"
    opt = next((r for r in job.results.values() if r.tool == "optimize_facility_locations"), None)
    opt_html = ""
    if opt is not None:
        s = opt.summary
        opt_html = (f"<h2>Optimization</h2><p>Objective <code>{s['objective']}</code>, p={s['p']}, radius {s['radius_m']:g} m. Demand: {html.escape(s['demand_proxy'])}.</p>"
                    f"{_table(opt.table)}<pre>{html.escape(json.dumps({k: s[k] for k in ('problem', 'before', 'greedy', 'cpsat', 'random_baseline')}, indent=2, default=str))}</pre>"
                    "<p><b>Assumptions:</b></p><ul>" + "".join(f"<li>{html.escape(a)}</li>" for a in s["assumptions"]) + "</ul>")
    ml = next((r for r in job.results.values() if r.tool == "classify_land_use"), None)
    ml_html = ""
    if ml is not None:
        ml_html = f"<h2>Validated predictive model</h2><pre>{html.escape(json.dumps(ml.summary, indent=2, default=str)[:3500])}</pre>"
    audit = None
    try:
        from .executor import Executor
        job.answer = Executor.compose_answer(job)
        audit = Executor.audit_claims(job)
    except Exception:  # noqa: BLE001
        pass
    gaps = next((r.summary.get("gaps", []) for r in job.results.values() if r.tool == "get_geographic_coverage"), [])
    layers = _layers(job, ctx)
    bb = ctx.area.bbox
    map_js = f"""
const L_=document.getElementById('map');
const m=L.map('map').fitBounds([[{bb[1]},{bb[0]}],[{bb[3]},{bb[2]}]]);
L.tileLayer('https://tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png',{{maxZoom:19,attribution:'© OpenStreetMap contributors; data © Overture Maps Foundation'}}).addTo(m);
const layers={json.dumps(layers)};const ctl={{}};
layers.forEach(l=>{{const g=L.geoJSON(l.geojson,{{style:{{color:l.color,weight:2,fillOpacity:.15}},pointToLayer:(f,ll)=>L.circleMarker(ll,{{radius:7,color:'#222',weight:1,fillColor:l.color,fillOpacity:.9}}),
 onEachFeature:(f,lay)=>lay.bindPopup('<pre style="max-width:320px;white-space:pre-wrap">'+JSON.stringify(f.properties,null,1)+'</pre>')}}).addTo(m);ctl[l.tool+' ('+l.id+')']=g;}});
L.control.layers(null,ctl).addTo(m);"""
    doc = f"""<!doctype html><meta charset="utf-8"><title>Geo-RiyadhAI evidence report {job.id}</title>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css">
<style>body{{font:15px/1.55 system-ui,sans-serif;max-width:1000px;margin:2rem auto;padding:0 1rem}}#map{{height:440px;border:1px solid #ccc}}table.t{{border-collapse:collapse;font-size:13px}}
table.t td,table.t th{{border:1px solid #ddd;padding:3px 7px}}code{{background:#f2f2f2;padding:1px 4px;font-size:12px}}pre{{background:#f7f7f7;padding:8px;overflow:auto;font-size:12px}}
blockquote{{border-left:3px solid #bbb;margin-left:0;padding-left:10px;color:#333}}.warn{{background:#fff4e5;border:1px solid #f0c36d;padding:8px 12px}}</style>
<h1>Evidence report</h1>
<p><b>Research question:</b> {html.escape(job.question)}<br><b>Job:</b> <code>{job.id}</code> · <b>Generated:</b> {time.strftime('%Y-%m-%d %H:%M:%S')} · Geo-RiyadhAI {__version__}</p>
<h2>Geographic coverage</h2><p>Area <b>{html.escape(ctx.area.name)}</b>, bbox (W,S,E,N) {tuple(round(x, 4) for x in bb)}. {len(ctx.area.layers['poi'])} POIs, {len(ctx.area.layers['road'])} drivable road segments, {len(ctx.area.layers.get('building', []))} buildings, {len(ctx.area.layers.get('district', []))} district polygons.</p>
<div class="warn"><b>Read before using:</b> {'; '.join(html.escape(g) for g in gaps) or 'see limitations'}. Suitability rankings and optimization results are analytical outputs on a candidate grid, not approvals, not buildable-land checks, and not predictions of commercial success.</div>
<h2>Datasets</h2><table class="t"><tr><th>Layer</th><th>Provider</th><th>Release</th><th>Licence</th><th>Retrieved</th></tr>{prov_rows}</table>
<h2>Map</h2><div id="map"></div>
<h2>Findings</h2>{facts_html}{expl}{opt_html}{ml_html}
<h2>Tables</h2>{tables}
<h2>Methodology (retrieved reference passages)</h2>{rag or '<p><i>none retrieved</i></p>'}
<h2>Data quality, uncertainty and limitations</h2><ul>
{''.join(f'<li>{html.escape(w)}</li>' for w in dict.fromkeys(job.warnings))}
<li>POI and building data are aggregated from open sources; completeness is unknown. POIs are points without entrances.</li><li>All distances are in metres (UTM zone {ctx.area.crs.to_epsg()}); network distances are undirected and are not travel times.</li></ul>
<h2>Claim audit</h2><p>Every number in the findings was checked against tool outputs: <code>{html.escape(json.dumps(audit))}</code></p>
<h2>Reproduction</h2><pre>{html.escape(json.dumps(job.plan.to_dict(), indent=2, default=str))}</pre>
<p>Python {platform.python_version()}; steps executed with the arguments recorded in the job trace (<code>data/analyses/jobs/{job.id}.json</code>).</p>
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script><script>{map_js}</script>"""
    out.write_text(doc, encoding="utf-8")
    return out
