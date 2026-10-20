"""Vetted analysis recipes: ready-made, reviewed analysis programs for question types that come up often.

A recipe is NOT a separate engine. It produces the same kind of artefact an LLM would (a plan and runnable Python), which then goes through the same
sandbox, observation, checks, plot and explanation as everything else. They make the agent useful with no LLM configured and give the LLM
worked examples. Each recipe prints one `SUMMARY {json}` line so the explanation is built from computed numbers only.
"""
from __future__ import annotations

import json
import re
from typing import Optional

AIRPORT_TOKENS = ["airport", "terminal", "مطار", "الصالة"]

HOMES_NEAR_TRANSIT_CODE = r'''
# Homes with the best metro connection to the airport and enough parking nearby.
import json
MIN_VEHICLES = {min_vehicles}      # parking capacity (vehicles) required within RADIUS_M
RADIUS_M = {radius_m}              # parking search radius
TOP = 25
WALK_DETOUR = 1.3                  # straight-line distance x 1.3 ~ walking distance (assumption, no pedestrian network)
WALK_KMH, METRO_KMH, WAIT_MIN = 5.0, 35.0, 5.0   # assumptions for the time estimate
AIRPORT_TOKENS = {airport_tokens}

b = layers["urbi_building"]
homes = in_scope(b[b["purpose_class"] == "residential"]).copy()
if len(homes) == 0:
    raise ValueError("no residential buildings in the selected area")
hp = homes.geometry.representative_point()
hx, hy = hp.x.to_numpy(), hp.y.to_numpy()

# ---- metro network: stations, lines (noded at crossings), distance along the lines to the nearest airport station
from scipy.spatial import cKDTree
from shapely.ops import unary_union
st = layers["metro_stations"].copy()
ln = layers["metro_lines"]
air = st[st["name"].str.contains("|".join(AIRPORT_TOKENS), case=False, na=False)]
if len(air) == 0:
    raise ValueError("no airport metro station found in the metro data")
merged = unary_union(list(ln.geometry))
segs = list(merged.geoms) if hasattr(merged, "geoms") else [merged]
G = nx.Graph()
for s in segs:
    c = [(round(x, 1), round(y, 1)) for x, y in s.coords]
    for p, q in zip(c[:-1], c[1:]):
        G.add_edge(p, q, weight=float(np.hypot(p[0] - q[0], p[1] - q[1])))
nodes = np.array(list(G.nodes))
nt = cKDTree(nodes)
for i, j in nt.query_pairs(30.0):          # lines cut at tile borders can be a few metres apart: join them
    p, q = tuple(nodes[i]), tuple(nodes[j])
    G.add_edge(p, q, weight=float(np.hypot(p[0] - q[0], p[1] - q[1])))
sxy = np.c_[st.geometry.x, st.geometry.y]
snap_d, snap_i = nt.query(sxy)
air_nodes = [tuple(nodes[snap_i[k]]) for k in range(len(st)) if st.index[k] in set(air.index)]
dist_to_air = nx.multi_source_dijkstra_path_length(G, air_nodes, weight="weight")
st["ride_km"] = [dist_to_air.get(tuple(nodes[i]), np.nan) / 1000.0 for i in snap_i]
st["snap_m"] = snap_d
usable = st[st["ride_km"].notna() & (st["snap_m"] < 400)]
if len(usable) == 0:
    raise ValueError("no station is connected to the airport along the metro lines")
ut = cKDTree(np.c_[usable.geometry.x, usable.geometry.y])
d_st, i_st = ut.query(np.c_[hx, hy])
homes["station"] = usable["name"].to_numpy()[i_st]
homes["walk_m"] = np.round(d_st * WALK_DETOUR, 0)
homes["ride_km"] = np.round(usable["ride_km"].to_numpy()[i_st], 1)
homes["est_minutes"] = np.round(homes["walk_m"] / 1000 / WALK_KMH * 60 + WAIT_MIN + homes["ride_km"] / METRO_KMH * 60, 0)

# ---- parking with enough reported capacity within RADIUS_M (unknown capacity is NOT counted)
pk = layers["parking"]
cap = pd.to_numeric(pk["capacity"], errors="coerce") if "capacity" in pk else pd.Series(np.nan, index=pk.index)
pkp = pk.geometry.representative_point()
pt = cKDTree(np.c_[pkp.x, pkp.y])
cap_known = cap.fillna(0).to_numpy()
unknown = cap.isna().to_numpy()
cap_sum, lots, unk = np.zeros(len(homes)), np.zeros(len(homes)), np.zeros(len(homes))
for k, idx in enumerate(pt.query_ball_point(np.c_[hx, hy], RADIUS_M)):
    if idx:
        cap_sum[k], lots[k], unk[k] = cap_known[idx].sum(), len(idx), unknown[idx].sum()
homes["parking_capacity"], homes["parking_lots"], homes["lots_capacity_unknown"] = cap_sum, lots, unk
homes["district"] = district_of(homes.assign(geometry=hp))

ok = homes[homes["parking_capacity"] >= MIN_VEHICLES].copy()
ok["geometry"] = hp[ok.index]
ok = ok.sort_values(["est_minutes", "walk_m"]).head(TOP)
ok.insert(0, "rank", range(1, len(ok) + 1))
cols = ["rank", "district", "station", "walk_m", "ride_km", "est_minutes", "parking_capacity", "parking_lots", "lots_capacity_unknown", "floors", "name", "geometry"]
result = gpd.GeoDataFrame(ok[[c for c in cols if c in ok.columns]], geometry="geometry", crs=crs)
summary = {{"residential_buildings": int(len(homes)), "with_enough_parking": int((homes["parking_capacity"] >= MIN_VEHICLES).sum()),
           "with_any_parking_lot": int((homes["parking_lots"] > 0).sum()), "max_capacity_within_radius": float(homes["parking_capacity"].max()),
           "shown": int(len(result)), "min_vehicles": MIN_VEHICLES, "radius_m": RADIUS_M,
           "best_minutes": float(result["est_minutes"].min()) if len(result) else None, "median_minutes_all_homes": float(homes["est_minutes"].median()),
           "best_walk_m": float(result["walk_m"].iloc[0]) if len(result) else None, "best_station": str(result["station"].iloc[0]) if len(result) else None,
           "best_ride_km": float(result["ride_km"].iloc[0]) if len(result) else None,
           "airport_stations": sorted(set(air["name"])), "stations_connected": int(len(usable)),
           "unknown_capacity_share": float(unknown.sum() / max(1, len(pk)))}}
print("SUMMARY " + json.dumps(summary, ensure_ascii=False))
notes.append("walking distance = straight line x 1.3 (no pedestrian network); times assume 5 km/h walking, 5 min wait, 35 km/h metro")
notes.append("parking capacity counts only lots with a reported capacity; lots without one are not counted")

# ---- figure: map + bar chart
fig, (ax, bx) = plt.subplots(1, 2, figsize=(13, 5.5), gridspec_kw={{"width_ratios": [1.3, 1]}})
if scope is not None:
    gpd.GeoSeries([scope], crs=crs).boundary.plot(ax=ax, color="#344054", linewidth=1)
homes.geometry.representative_point().plot(ax=ax, color="#d0d5dd", markersize=2)
if len(result):
    result.plot(ax=ax, column="est_minutes", cmap="viridis_r", markersize=60, legend=True, legend_kwds={{"label": "est. minutes to airport"}}, zorder=4)
usable.plot(ax=ax, color="#7a5af8", marker="s", markersize=30, zorder=3, label="metro station")
air.plot(ax=ax, color="#f04438", marker="*", markersize=140, zorder=5, label="airport station")
pkp.plot(ax=ax, color="#12b76a", markersize=3, alpha=.5, label="parking")
if scope is not None:
    minx, miny, maxx, maxy = scope.buffer(800).bounds
    ax.set_xlim(minx, maxx); ax.set_ylim(miny, maxy)
ax.set_title("Best homes by metro connection to the airport"); ax.set_axis_off(); ax.legend(loc="lower left", fontsize=8)
if len(result):
    top = result.head(12)
    bx.barh([f"#{{r}} {{d or ''}}"[:24] for r, d in zip(top["rank"], top["district"])][::-1], top["est_minutes"][::-1], color="#2e90fa")
    bx.set_xlabel("estimated door-to-airport minutes"); bx.set_title("Fastest homes that also meet the parking rule")
else:
    bx.text(.5, .5, "no home meets the parking rule", ha="center"); bx.set_axis_off()
plt.tight_layout()
'''


def match(question: str) -> Optional[dict]:
    """Return recipe parameters if the question is one this recipe handles, else None."""
    low = question.lower()
    homes = re.search(r"\b(house|houses|home|homes|villa|villas|apartment|apartments|residential|residence|residences)\b", low)
    transit = re.search(r"\b(metro|station|stations|train|transit)\b", low)
    if not (homes and transit):
        return None
    mv = re.search(r"(\d+)\s*(?:vehicles?|cars?|spaces?|spots?|slots?|parking)", low)
    rad = re.search(r"within\s+(\d+(?:\.\d+)?)\s*(km|kilomet\w*|m|met\w*)\b", low)
    r = 1000.0
    if rad:
        r = float(rad.group(1)) * (1000.0 if rad.group(2).startswith("k") else 1.0)
    wants_parking = bool(re.search(r"parking|park", low))
    return {"recipe": "homes_near_transit", "min_vehicles": int(mv.group(1)) if mv else (10 if wants_parking else 0), "radius_m": r,
            "airport": bool(re.search(r"airport|مطار", low))}


def render(params: dict) -> tuple[dict, str]:
    plan = {"interpretation": f"Find residential buildings in the selected area, rank them by estimated door-to-airport time using the metro (walk to the nearest station + metro distance along the lines), "
                              f"and keep only those with at least {params['min_vehicles']} vehicles of known parking capacity within {params['radius_m'] / 1000:g} km.",
            "steps": ["select residential buildings (Urbi purpose) inside the area", "build the metro network from the line geometry and find the distance along it from every station to the airport terminals",
                      "assign each home its nearest connected station and estimate walk + wait + ride time", f"sum reported parking capacity within {params['radius_m']:g} m of each home",
                      "keep homes meeting the parking rule, rank by time, map and chart the best"],
            "assumptions": ["no pedestrian network: walking = straight line x 1.3", "metro speed 35 km/h and 5 min wait are assumptions, not timetable data", "parking lots without a reported capacity are not counted"],
            "layers": ["urbi_building", "metro_stations", "metro_lines", "parking", "district"]}
    code = HOMES_NEAR_TRANSIT_CODE.format(min_vehicles=params["min_vehicles"], radius_m=params["radius_m"], airport_tokens=json.dumps(AIRPORT_TOKENS, ensure_ascii=False))
    return plan, code
