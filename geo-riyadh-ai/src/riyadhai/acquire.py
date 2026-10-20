"""Data acquisition the analysis agent can request, behind hard guardrails. The model chooses WHAT is missing; this code decides whether it may be fetched.

Tools (typed; the model cannot write download code):
  list_sources   which sources exist, what each offers, and whether it is authorized and usable right now
  estimate       cost of fetching layers for the selected area: cached or live, expected seconds, area size
  fetch          fetch layers (Overture open data through the GeoAcquirer client): cache first; LIVE downloads need the user's approval
  (verify runs automatically after every fetch)
Guardrails: only whitelisted layers; only authorized/open providers (Urbi and Suhail stay disabled here unless the environment records an authorization AND
an adapter is wired in); area <= 0.1 deg2; at most 3 fetches per analysis; live downloads raise NeedsApproval instead of running; every fetch is verified
(row count, coverage of the area, empty/invalid geometry, missing key fields) and the report is part of what the model observes and what the user sees.
Credentials never reach the model or the sandbox.
"""
from __future__ import annotations

from typing import Optional

import geopandas as gpd
from shapely.geometry import box

ALLOWED_LAYERS = ("poi", "road", "connector", "building", "district")
MAX_BBOX_DEG2 = 0.1
MAX_FETCHES = 3
LIVE_SECONDS_PER_LAYER = 120   # registry estimate for Overture via DuckDB/S3


class NeedsApproval(Exception):
    def __init__(self, why: str, action: dict):
        super().__init__(why)
        self.why, self.action = why, action


class AcquisitionError(RuntimeError):
    pass


class AcquisitionTools:
    def __init__(self, client, approved: bool = False, registry: Optional[dict] = None):
        self.client, self.approved = client, approved
        self.fetches = 0
        self._registry = registry

    @property
    def registry(self) -> dict:
        if self._registry is None:
            from geoacquirer.providers.registry import load_registry
            self._registry = load_registry()
        return self._registry

    # ------------------------------------------------------------------ tools
    def list_sources(self) -> dict:
        from geoacquirer.providers.registry import is_authorized
        out = {}
        for pid, e in self.registry.items():
            ok, ref = is_authorized(pid, e)
            usable = ok and pid == "overture"   # the only provider this tool fetches from
            out[pid] = {"authorized": bool(ok), "usable_by_this_tool": usable, "entity_types": e.get("entity_types", []), "licence": str(e.get("allowed_usage", ""))[:200],
                        "limits": str(e.get("limits", ""))[:200], "why_not": None if usable else ("not authorized in this session" if not ok else "no adapter for the fetch tool (use the prepared city data)")}
        return out

    def _check(self, layers, bbox):
        bad = [l for l in layers if l not in ALLOWED_LAYERS]
        if bad or not layers:
            raise AcquisitionError(f"layers must be a non-empty subset of {list(ALLOWED_LAYERS)}; got {layers}")
        w, s, e, n = bbox
        if not (w < e and s < n):
            raise AcquisitionError("invalid bbox")
        if (e - w) * (n - s) > MAX_BBOX_DEG2:
            raise AcquisitionError(f"the area is too large to fetch in one go ({(e - w) * (n - s):.3f} deg2 > {MAX_BBOX_DEG2}); use a smaller area")

    def estimate(self, layers: list, bbox: tuple) -> dict:
        self._check(layers, bbox)
        cached = self.client.cached_layers(bbox, tuple(layers))
        live = [l for l in layers if not cached.get(l)]
        return {"layers": {l: {"cached": bool(cached.get(l)), "est_seconds": 0 if cached.get(l) else LIVE_SECONDS_PER_LAYER, "est_requests": 0 if cached.get(l) else 1} for l in layers},
                "needs_live_download": bool(live), "est_total_seconds": LIVE_SECONDS_PER_LAYER * len(live), "bbox_deg2": round((bbox[2] - bbox[0]) * (bbox[3] - bbox[1]), 4),
                "source": "overture (open data; ODbL/CDLA, attribution required)", "approval_needed": bool(live) and not self.approved}

    def fetch(self, layers: list, bbox: tuple) -> tuple[dict, dict]:
        self._check(layers, bbox)
        if self.fetches >= MAX_FETCHES:
            raise AcquisitionError(f"fetch budget used up ({MAX_FETCHES} per analysis)")
        est = self.estimate(layers, bbox)
        if est["needs_live_download"] and not self.approved:
            raise NeedsApproval(f"download {', '.join(l for l, v in est['layers'].items() if not v['cached'])} from Overture for this area (about {est['est_total_seconds'] // 60 + 1} minutes)",
                                {"tool": "fetch", "layers": layers, "bbox": list(bbox)})
        self.fetches += 1
        res = self.client.fetch_layers(bbox, tuple(layers), allow_live=self.approved)
        reports = {l: self.verify(l, res.layers[l], bbox) for l in layers}
        return res.layers, {"from_cache": res.from_cache, "seconds": round(res.seconds, 1), "warnings": res.warnings[:5], "verification": reports, "provenance": {l: res.provenance.get(l, {}).get("dataset_version") for l in layers}}

    def verify(self, layer: str, gdf: gpd.GeoDataFrame, bbox: tuple) -> dict:
        """Completeness / correctness checks the model and the user can see."""
        rep = {"rows": int(len(gdf)), "ok": True, "issues": []}
        if len(gdf) == 0:
            rep["ok"] = False
            rep["issues"].append("no features returned: either nothing is mapped here or the source is incomplete; do not treat as 'zero'")
            return rep
        inv = int((~gdf.geometry.is_valid).sum()) if len(gdf) else 0
        if inv:
            rep["issues"].append(f"{inv} invalid geometries")
        nul = int(gdf.geometry.isna().sum() + gdf.geometry.is_empty.sum())
        if nul:
            rep["issues"].append(f"{nul} empty/missing geometries")
        b = gdf.total_bounds
        w, s, e, n = bbox
        inside = box(w, s, e, n).intersection(box(*b)).area / max(1e-12, box(w, s, e, n).area)
        rep["extent_covers_bbox"] = round(float(inside), 2)
        if inside < 0.5:
            rep["issues"].append(f"features cover only {inside * 100:.0f}% of the requested box: coverage may be partial")
        for col in {"poi": ["name", "category_path"], "building": ["height"], "road": ["class"]}.get(layer, []):
            if col in gdf:
                share = float(gdf[col].isna().mean())
                rep[f"missing_{col}"] = round(share, 2)
                if share > 0.8:
                    rep["issues"].append(f"{col} is missing for {share * 100:.0f}% of rows")
        rep["ok"] = not rep["issues"]
        return rep

    # ------------------------------------------------------------------ dispatcher used by the agent
    def call(self, tool: str, args: dict, default_bbox: tuple) -> tuple[dict, dict]:
        """Returns (observation dict, {layer name: GeoDataFrame in WGS84} to add to the analysis)."""
        if tool == "list_sources":
            return {"sources": self.list_sources()}, {}
        layers = list(args.get("layers", []))
        bbox = tuple(args.get("bbox") or default_bbox)
        if tool == "estimate":
            return self.estimate(layers, bbox), {}
        if tool == "fetch":
            data, report = self.fetch(layers, bbox)
            return {"fetched": {l: len(g) for l, g in data.items()}, **report, "available_as": [f"layers['acq_{l}']" for l in data]}, data
        raise AcquisitionError(f"unknown acquisition tool {tool!r}; use list_sources, estimate or fetch")
