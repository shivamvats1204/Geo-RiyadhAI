"""Worker process for sandbox.run_code (executed with `python -I`; never imported by the app).

Namespace given to the analysis code:
  layers   dict name -> GeoDataFrame (metric CRS, clipped to the selected area)
  crs      the metric CRS (units are metres)
  scope    shapely polygon of the selected area (metric CRS), or None
  pd, np, gpd, plt, shapely, nx (networkx), entities (places named in the question -> metric x/y), helpers: nearest_dist, count_within, district_of, in_scope, to_wgs84,
  and, when a model is configured, llm_query / llm_map (budgeted model calls, serviced by the parent process)
The code must put its answer in `result` (DataFrame / GeoDataFrame / dict / number / string); figures drawn with plt are saved automatically.
"""
import builtins
import io
import json
import sys
import traceback
from pathlib import Path

work = Path(sys.argv[1])
meta = json.loads((work / "meta.json").read_text(encoding="utf-8"))
ALLOWED = set(meta["allowed"])
status = {"ok": False}


def finish():
    (work / "status.json").write_text(json.dumps(status, default=str), encoding="utf-8")


try:
    import numpy as np
    import pandas as pd
    import geopandas as gpd
    import shapely
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import networkx as nx
    from shapely import wkt as _wkt

    layers = {}
    for p in sorted(Path(meta["layers_dir"]).glob("*.parquet")):
        layers[p.stem] = gpd.read_parquet(p)
    crs = meta["crs"]
    scope = _wkt.loads(meta["scope_wkt"]) if meta.get("scope_wkt") else None

    def nearest_dist(a, b):
        """Distance in metres from each geometry of `a` to the nearest geometry of `b` (both GeoDataFrames/GeoSeries in the metric CRS); NaN if b is empty."""
        ag = a.geometry if hasattr(a, "geometry") else a
        bg = b.geometry if hasattr(b, "geometry") else b
        out = np.full(len(ag), np.nan)
        if len(bg) == 0 or len(ag) == 0:
            return pd.Series(out, index=ag.index)
        tree = shapely.STRtree(bg.values)
        idx, d = tree.query_nearest(ag.values, return_distance=True, all_matches=False)
        out[idx[0]] = d
        return pd.Series(out, index=ag.index)

    def count_within(a, b, distance_m, weight=None):
        """For each geometry of `a`: how many geometries of `b` lie within distance_m (or the sum of column `weight` of b)."""
        ag = a.geometry if hasattr(a, "geometry") else a
        res = np.zeros(len(ag))
        if len(b) == 0 or len(ag) == 0:
            return pd.Series(res, index=ag.index)
        tree = shapely.STRtree(b.geometry.values)
        ai, bi = tree.query(ag.values, predicate="dwithin", distance=distance_m)
        w = np.ones(len(b)) if weight is None else pd.to_numeric(b[weight], errors="coerce").fillna(0).to_numpy()
        np.add.at(res, ai, w[bi])
        return pd.Series(res, index=ag.index)

    def district_of(points, name_col="name"):
        """Name of the district containing each point (uses layers['district'])."""
        d = layers.get("district")
        if d is None or len(d) == 0:
            return pd.Series([None] * len(points), index=points.index)
        j = gpd.sjoin(gpd.GeoDataFrame(geometry=points.geometry.values, crs=points.crs), d[[name_col, "geometry"]], how="left", predicate="within")
        j = j[~j.index.duplicated()]
        return pd.Series(j[name_col].to_numpy(), index=points.index)

    def in_scope(gdf):
        """Keep only the features whose representative point lies inside the selected area (layers carry a margin around it so distances near the edge are right)."""
        if scope is None or len(gdf) == 0:
            return gdf
        return gdf[gdf.geometry.representative_point().within(scope).to_numpy()]

    def to_wgs84(gdf):
        return gdf.to_crs(4326)

    import time as _time
    _rpc_dir = work / "rpc"
    _n = [0]

    def _call_model(prompt):
        """Ask the parent process for a model completion (file handshake; this process has no network and no keys)."""
        _n[0] += 1
        req = _rpc_dir / f"req_{_n[0]:04d}.json"
        tmp = req.with_suffix(".tmp")
        tmp.write_text(json.dumps({"prompt": str(prompt), "opts": {}}), encoding="utf-8")
        tmp.replace(req)
        resp = _rpc_dir / f"resp_{_n[0]:04d}.json"
        while not resp.exists():
            _time.sleep(0.05)
        d = json.loads(resp.read_text(encoding="utf-8"))
        if not d["ok"]:
            raise RuntimeError(d["error"])
        return d["text"]

    def llm_query(prompt):
        """Ask a language model ONE question about text you pass in. Returns a string. For reading/classifying/summarising text; never for counting, distances or joins.
        Its answers are model judgements, not measurements. Budgeted: a limited number of calls per analysis."""
        return _call_model(prompt)

    def llm_map(items, instruction, batch=25):
        """Apply `instruction` to a list of short texts with a language model, `batch` items per call; returns a list with one result per item (None when unknown)."""
        items = list(items)
        out = []
        for i in range(0, len(items), batch):
            chunk = items[i:i + batch]
            for attempt in range(2):
                txt = _call_model(instruction + f"\nReturn ONLY a JSON list with exactly {len(chunk)} elements, one per input, in the same order.\nINPUTS:\n" + json.dumps(chunk, ensure_ascii=False))
                a, b = txt.find("["), txt.rfind("]")
                try:
                    arr = json.loads(txt[a:b + 1])
                except ValueError:
                    arr = None
                if isinstance(arr, list) and len(arr) == len(chunk):
                    out.extend(arr)
                    break
            else:
                raise ValueError(f"the model did not return a JSON list of {len(chunk)} items for batch starting at {i}")
        return out

    from shapely.geometry import Point as _Pt

    class _Place(dict):
        """A resolved place: dict access (['x'], ['kind'], ['district']) and attributes (.x, .y metric metres, .geometry a shapely Point)."""
        x = property(lambda self: self["x"])
        y = property(lambda self: self["y"])
        kind = property(lambda self: self.get("kind"))
        geometry = property(lambda self: _Pt(self["x"], self["y"]))

    class _CI(dict):
        """Case-insensitive lookup for the resolved places (the model may write 'Kingdom Centre' or 'kingdom centre')."""
        def __init__(self, d):
            super().__init__({str(k).lower(): _Place(v) for k, v in d.items()})

        def __getitem__(self, k):
            try:
                return super().__getitem__(str(k).lower())
            except KeyError:
                raise KeyError(f"{k!r}; available places: {list(self.keys())}") from None

        def get(self, k, default=None):
            return super().get(str(k).lower(), default)

        def __contains__(self, k):
            return super().__contains__(str(k).lower())

    real_import = builtins.__import__

    def safe_import(name, globals=None, locals=None, fromlist=(), level=0):
        if level or name.split(".")[0] not in ALLOWED:
            raise ImportError(f"import of '{name}' is not allowed")
        return real_import(name, globals, locals, fromlist, level)

    BLOCKED = {"open", "exec", "eval", "compile", "input", "breakpoint", "globals", "locals", "vars", "getattr", "setattr", "delattr", "memoryview", "exit", "quit", "help", "dir"}
    safe_builtins = {k: v for k, v in vars(builtins).items() if k not in BLOCKED and k != "__import__"}
    safe_builtins["__import__"] = safe_import
    ns = {"__builtins__": safe_builtins, "layers": layers, "crs": crs, "scope": scope, "pd": pd, "np": np, "gpd": gpd, "plt": plt, "shapely": shapely, "nx": nx,
          "nearest_dist": nearest_dist, "count_within": count_within, "district_of": district_of, "in_scope": in_scope, "to_wgs84": to_wgs84, "entities": _CI(meta.get("entities", {})), "notes": [], "result": None}
    if meta.get("llm_enabled"):
        ns["llm_query"], ns["llm_map"] = llm_query, llm_map
    code = (work / "code.py").read_text(encoding="utf-8")
    buf = io.StringIO()
    old = sys.stdout
    sys.stdout = buf
    try:
        exec(compile(code, "analysis.py", "exec"), ns)  # noqa: S102  (validated by sandbox.validate_code before the process was started)
    finally:
        sys.stdout = old
    status["stdout"] = buf.getvalue()[-8000:]
    res = ns.get("result")
    if res is None:
        status["error"] = "the code did not set `result`"
    elif isinstance(res, gpd.GeoDataFrame) and "geometry" in res and len(res):
        res.to_parquet(work / "result.parquet"); status.update(result_kind="geodataframe", result_path=str(work / "result.parquet"), ok=True)
    elif isinstance(res, pd.DataFrame):
        res.to_parquet(work / "result.parquet") if not isinstance(res, gpd.GeoDataFrame) else pd.DataFrame(res.drop(columns="geometry", errors="ignore")).to_parquet(work / "result.parquet")
        status.update(result_kind="dataframe", result_path=str(work / "result.parquet"), ok=True)
    elif isinstance(res, pd.Series):
        res.to_frame(res.name or "value").reset_index().to_parquet(work / "result.parquet"); status.update(result_kind="dataframe", result_path=str(work / "result.parquet"), ok=True)
    else:
        (work / "result.json").write_text(json.dumps(res, default=lambda o: o.item() if hasattr(o, "item") else str(o)), encoding="utf-8")
        status.update(result_kind="json", result_path=str(work / "result.json"), ok=True)
    figs = []
    for i, n in enumerate(plt.get_fignums()[:3]):
        f = work / f"fig{i}.png"
        plt.figure(n).savefig(f, dpi=110, bbox_inches="tight")
        figs.append(str(f))
    status["figures"] = figs
    status["notes"] = [str(x) for x in ns.get("notes", [])][:10]
except BaseException:  # noqa: BLE001
    tb = traceback.format_exc().splitlines()
    status["ok"] = False
    status["error"] = "\n".join(tb[-12:])
finally:
    finish()
