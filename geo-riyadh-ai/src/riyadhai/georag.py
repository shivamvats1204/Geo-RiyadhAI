"""Geo-RAG: retrieval that grounds the code-writing agent. Retrieval FINDS; code COMPUTES.

Four indexes (all local; TF-IDF / character n-grams, no model or key needed):
  catalogue   what each layer and column means, its source, licence, units and pitfalls (hand-written notes + live schema stats)
  entities    named places (districts, neighbourhoods, attractions, metro stations, streets, POIs) in English and Arabic with fuzzy matching, resolved to
              coordinates, so "near Kingdom Centre" or "المربع" becomes a real location the code can use
  examples    verified past analyses (question, plan, code) retrieved as few-shot examples; only analyses that ran, passed the checks and
              had no untraceable numbers are remembered
  docs        the reference notes (zoning fields, methods, data sources)
It never answers numeric questions itself: counts, distances and rankings come from running code on the data.
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import geopandas as gpd
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# What the columns mean and where they mislead. Keyed (layer, column); layer-level notes use column None.
CATALOGUE_NOTES = {
    ("urbi_building", None): "Urbi/2GIS building footprints (polygons) for all of Riyadh. Source: Urbi catalogue, used under the user's stated permission (no licence document seen). `name` is usually the street address, not a business name.",
    ("urbi_building", "purpose_class"): "Coarse use of the building derived from Urbi's Arabic/English `purpose` text by keyword mapping: residential, commercial, industrial, public, education, religious, health, other. Not an official land-use class; use it for houses/homes/apartments (residential) vs shops/offices (commercial).",
    ("urbi_building", "purpose"): "Raw Urbi purpose text (e.g. 'Residential', 'Shop', 'Mosque'); many buildings have none.",
    ("urbi_building", "floors"): "Number of floors above ground as recorded by Urbi (numeric, often missing).",
    ("parking", None): "Parking lots. Overture POIs (points) plus Urbi parking (polygons). Use geometry.representative_point() for distances.",
    ("parking", "capacity"): "Number of vehicles, operator-supplied, often missing (NaN). Lots without capacity must be reported as unknown, never counted as zero or as enough.",
    ("parking", "access"): "Access type as recorded by Urbi (e.g. public/private/customers); missing for Overture points.",
    ("parking", "is_paid"): "Whether parking is paid, where recorded.",
    ("poi", None): "Overture Maps places (points): shops, restaurants, clinics, banks... `category_path` is a list from broad to specific, e.g. ['shopping', 'grocery_store']. Coverage is uneven; places can be missing or stale.",
    ("poi", "category_path"): "List of category tokens, broad to specific. Test membership of a token ('pharmacy', 'supermarket'), do not compare the whole list to a string.",
    ("metro_stations", None): "Riyadh Metro stations from Suhail vector tiles (92 stations, Arabic names). The three airport stops are named like 'محطة الصالة 5', 'محطة الصالة 3-4', 'محطة الصالة 1-2' (terminal = الصالة).",
    ("metro_lines", None): "Riyadh Metro lines (6, one per colour) from Suhail vector tiles. Lines connect only at interchange crossings; node the linework (shapely unary_union) before building a graph. No timetable or frequency data exists.",
    ("metro", None): "Urbi metro objects (platforms/entrances/stops, polygons); `route_type` is 'metro' or 'bus'. Names are Latin-script (e.g. 'Airport T5 18').",
    ("parcel", None): "Suhail land parcels (polygons) with first zoning rule: `landuse` (mixed/commercial/industrial/residential/other/unknown, keyword mapping of the Arabic text). Coverage has a gap at the outer edge of the city.",
    ("parcel", "maxBuildingHeight"): "Maximum building height from the zoning rule; the unit (metres or floors) is not stated by the source: confirm before use.",
    ("parcel", "maxBuildingCoefficient"): "Maximum floor-area ratio (building coefficient) from the zoning rule.",
    ("district", None): "Urbi districts (polygons; English `name`, `name_ar`). Some names repeat for different polygons; use `district_id`.",
    ("street", None): "Urbi street objects (polygons/strips) with names; not a routable network. For routing use the road network tools, not this layer.",
    ("attraction", None): "Urbi attractions/landmarks (points and polygons), named; includes minor objects (benches, light poles) in `subtype`.",
    ("junction", None): "Road junctions derived from Overture connectors: kind four_way / t_junction / highway_interchange. Divided carriageways are merged; grade-separated crossings are not junctions.",
    ("road", None): "Overture road segments (lines) with `class` (motorway, trunk, primary, ...). Undirected; no speeds.",
}
SOURCE_NOTES = {
    "urbi": "Urbi/2GIS catalogue: buildings, parking, streets, attractions, metro, administrative areas. Authorization required (recorded as user-stated permission).",
    "suhail": "Suhail: land parcels, zoning rules, neighbourhood polygons, metro lines and stations (vector tiles). Authorization required.",
    "overture": "Overture Maps (open): places, roads, connectors, buildings, divisions. Licences vary per record (CDLA / ODbL); attribution required.",
    "osm": "OpenStreetMap (open, ODbL): used through Overture and as the street basemap.",
}


@dataclass
class Retrieval:
    catalogue: list = field(default_factory=list)   # [{"key", "text"}]
    entities: list = field(default_factory=list)    # [{"name","kind","x","y","district","score"}]  x,y in the area's metric CRS
    examples: list = field(default_factory=list)    # [{"question","plan","code","score"}]
    docs: list = field(default_factory=list)        # [{"cite","text"}]

    def prompt_text(self, budget: int = 5200) -> str:
        parts = []
        if self.catalogue:
            parts.append("DATA NOTES (meaning, units, pitfalls):\n" + "\n".join(f"- {c['key']}: {c['text']}" for c in self.catalogue))
        if self.entities:
            parts.append("PLACES NAMED IN THE QUESTION (resolved; also available as the dict `entities`, metric x/y):\n" +
                         "\n".join(f"- {e['name']} ({e['kind']}{', ' + e['district'] if e.get('district') else ''}) at x={e['x']:.0f}, y={e['y']:.0f}  [match {e['score']:.2f}]" for e in self.entities))
        if self.docs:
            parts.append("REFERENCE NOTES:\n" + "\n".join(f"- {d['cite']} {d['text'][:260]}" for d in self.docs))
        if self.examples:
            parts.append("SIMILAR PAST ANALYSES THAT RAN AND PASSED THEIR CHECKS (adapt, do not copy blindly):\n" +
                         "\n\n".join(f"Q: {e['question']}\n```python\n{e['code'][:1500]}\n```" for e in self.examples))
        return "\n\n".join(parts)[:budget]


def _norm(s: str) -> str:
    s = str(s or "").lower()
    s = re.sub(r"[ً-ْـ]", "", s)                       # Arabic diacritics, tatweel
    s = s.translate(str.maketrans({"أ": "ا", "إ": "ا", "آ": "ا", "ى": "ي", "ة": "ه"}))
    s = re.sub(r"\b(al|el|ال)[\s\-]?", "", s)                         # the article is noise for matching
    return re.sub(r"[^\w\s]", " ", s).strip()


_STOP = {"the", "and", "with", "that", "has", "have", "for", "from", "near", "best", "find", "show", "give", "all", "any", "are", "within", "least", "than", "which", "where", "what", "how", "many", "much", "most",
         "residential", "houses", "house", "homes", "home", "parking", "vehicles", "metro", "airport", "airports", "connection", "via", "km", "shops", "shop", "district", "area", "city", "per", "each", "me", "those",
         "can", "get", "into", "of", "in", "to", "at", "on", "metre", "metres", "meter", "meters", "m", "km", "kilometre", "kilometres", "kilometer", "kilometers", "minute", "minutes", "min", "hour", "hours", "sqm", "percent", "number", "count", "total", "average", "mean", "median", "many", "list", "name", "names", "example", "examples", "look", "looks", "like", "judging", "give", "points", "interest", "buildings", "building", "places", "place", "connectors", "connector", "layer", "mapped", "use", "using", "a", "an", "is", "by", "or", "around", "close", "about", "this", "these", "their", "its", "it", "i", "my", "do", "does", "should", "would", "could"}
_GENERIC_NAMES = {_norm(x) for x in ("park", "parking", "school", "mall", "mosque", "hospital", "clinic", "bank", "street", "road", "station", "bus station", "small park", "playground", "garden", "market", "restaurant", "cafe", "shop", "pharmacy", "gas station", "monument", "fountain")}
# English -> Arabic for common place-name words, so English questions reach Arabic-only names (metro stations are Arabic only)
_BILINGUAL = {"ministry": "وزارة", "finance": "المالية", "interior": "الداخلية", "education": "التعليم", "health": "الصحة", "station": "محطة", "terminal": "الصالة", "airport": "مطار", "road": "طريق",
              "street": "شارع", "king": "الملك", "university": "جامعة", "hospital": "مستشفى", "mall": "مول", "park": "حديقة", "museum": "متحف", "national": "الوطني", "square": "ميدان", "bank": "بنك",
              "financial": "المالي", "center": "المركز", "centre": "المركز", "industrial": "الصناعية", "city": "المدينة", "technology": "التقنية", "salman": "سلمان", "abdulaziz": "عبدالعزيز", "fahd": "فهد"}


def _to_arabic(phrase: str) -> str:
    parts = [_BILINGUAL.get(w.lower(), w) for w in phrase.split()]
    return " ".join(parts) if any(p != w for p, w in zip(parts, phrase.split())) else phrase


class GeoRAG:
    def __init__(self, example_path: Optional[Path] = None, kb=None):
        self.entities = pd.DataFrame(columns=["name", "name_ar", "kind", "lon", "lat", "district"])
        self._keys, self._owner = [], np.array([], dtype=int)
        self._ent_vec = None
        self._ent_mat = None
        self._cat_keys, self._cat_texts = [], []
        self._cat_vec = self._cat_mat = None
        self.example_path = Path(example_path) if example_path else None
        self.examples: list[dict] = []
        self.kb = kb
        if self.example_path and self.example_path.exists():
            self.examples = [json.loads(l) for l in self.example_path.read_text(encoding="utf-8").splitlines() if l.strip()]
        self._ex_vec = self._ex_mat = None
        self._build_catalogue()
        self._fit_examples()

    # ------------------------------------------------------------------ catalogue
    def _build_catalogue(self, schemas: Optional[dict] = None):
        keys, texts = [], []
        for (layer, col), txt in CATALOGUE_NOTES.items():
            keys.append(f"layers['{layer}']" + (f".{col}" if col else ""))
            texts.append(f"{layer} {col or ''} {txt}")
        for k, v in SOURCE_NOTES.items():
            keys.append(f"source {k}")
            texts.append(f"{k} source licence authorization {v}")
        for key, txt in (schemas or {}).items():
            keys.append(key)
            texts.append(txt)
        self._cat_keys, self._cat_texts = keys, texts
        self._cat_vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, token_pattern=r"(?u)\b\w[\w']+\b")
        self._cat_mat = self._cat_vec.fit_transform(texts)

    def add_schema_stats(self, layers: dict):
        """Add live per-column notes (dtype, null share, examples) for the layers of the current area."""
        extra = {}
        for name, g in layers.items():
            for c in g.columns:
                if c == "geometry":
                    continue
                s = g[c]
                ex = ""
                nn = s.dropna()
                if len(nn) and not isinstance(nn.iloc[0], (list, np.ndarray, dict)):
                    ex = " examples: " + ", ".join(nn.astype(str).str.slice(0, 20).drop_duplicates().head(3))
                extra[f"layers['{name}'].{c} (live)"] = f"{name} {c} column dtype {s.dtype}, {s.isna().mean() * 100:.0f}% missing.{ex}"
        self._build_catalogue(extra)

    # ------------------------------------------------------------------ entities
    def set_entities(self, frame: pd.DataFrame):
        """frame columns: name, name_ar (optional), kind, lon, lat, district (optional)."""
        f = frame.copy()
        for c in ("name_ar", "district"):
            if c not in f:
                f[c] = None
        f = f.dropna(subset=["lon", "lat"]).reset_index(drop=True)
        f = f.reset_index(drop=True)
        keys, owner = [], []
        for i, (a, b) in enumerate(zip(f["name"], f["name_ar"])):
            for v in (a, b):  # one index entry per language so an English query is not diluted by the Arabic name (and vice versa)
                k = _norm(v) if isinstance(v, str) else ""
                if len(k) > 1:
                    keys.append(k)
                    owner.append(i)
        self.entities = f
        self._keys, self._owner = keys, np.array(owner, dtype=int)
        self._ent_vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), sublinear_tf=True)
        self._ent_mat = self._ent_vec.fit_transform(keys)

    def add_entities(self, frame: pd.DataFrame):
        self.set_entities(pd.concat([self.entities, frame], ignore_index=True).drop_duplicates(subset=["name", "name_ar", "kind", "lon", "lat"]))

    KIND_PRIOR = {"urbi district": 0.03, "suhail neighborhood": 0.02, "metro station": 0.03, "attraction": 0.01}

    def find_entities(self, question: str, k: int = 4, min_score: float = 0.6) -> list[dict]:
        """Named places mentioned in the question: word n-grams of the question matched (character n-gram cosine, length-aware) against the entity index.
        English words with a known Arabic equivalent are also tried in Arabic (ministry of finance -> وزارة المالية), because many names exist only in Arabic."""
        if self._ent_vec is None or self.entities.empty:
            return []
        words = [w for w in re.findall(r"[\w؀-ۿ']+", question) if len(w) > 1 or w.isdigit()]
        phrases = set()
        for n in (1, 2, 3, 4):
            for i in range(len(words) - n + 1):
                chunk = words[i:i + n]
                if all(re.fullmatch(r"[\d.,]+", w) for w in chunk) or chunk[0].lower() in _STOP or chunk[-1].lower() in _STOP or (n == 1 and (len(chunk[0]) < 4 or chunk[0].lower() in _BILINGUAL)):
                    continue
                phrases.add(" ".join(chunk))
        variants = []
        for ph in phrases:
            variants.append((ph, ph))
            ar = _to_arabic(ph)
            if ar != ph:
                variants.append((ph, ar))
        out: dict = {}
        for ph, v in variants:
            nv = _norm(v)
            if len(nv) < 3:
                continue
            sims = linear_kernel(self._ent_vec.transform([nv]), self._ent_mat).ravel()
            for j in np.argsort(-sims)[:3]:
                key = self._keys[j]
                ratio = min(len(key), len(nv)) / max(len(key), len(nv))
                sc = float(sims[j]) * (0.45 + 0.55 * ratio)
                if key in _GENERIC_NAMES:   # an entity called just 'park' or 'school' says nothing about WHICH one the user means
                    continue
                if key == nv:
                    sc = 1.0
                r = self.entities.iloc[self._owner[j]]
                sc += self.KIND_PRIOR.get(r["kind"], 0.0)
                if sc < min_score:
                    continue
                nm = r["name"] if isinstance(r["name"], str) and r["name"] else r["name_ar"]
                ident = (str(nm), r["kind"], round(float(r["lon"]), 5), round(float(r["lat"]), 5))
                if ident not in out or sc > out[ident]["score"]:
                    out[ident] = {"name": str(nm), "name_ar": r["name_ar"] if isinstance(r["name_ar"], str) else None, "kind": r["kind"], "lon": float(r["lon"]), "lat": float(r["lat"]),
                                  "district": r["district"] if isinstance(r["district"], str) else None, "score": min(sc, 1.0), "matched_phrase": ph}
        return sorted(out.values(), key=lambda e: -e["score"])[:k]

    # ------------------------------------------------------------------ examples (verified past analyses)
    def _fit_examples(self):
        if self.examples:
            self._ex_vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)
            self._ex_mat = self._ex_vec.fit_transform([e["question"] for e in self.examples])
        else:
            self._ex_vec = self._ex_mat = None

    def remember(self, question: str, plan: dict, code: str):
        """Store an analysis that ran, passed its checks and had no untraceable numbers. Near-duplicate questions replace the older entry."""
        rec = {"question": question.strip(), "plan": plan, "code": code, "t": int(time.time())}
        self.examples = [e for e in self.examples if _norm(e["question"]) != _norm(question)] + [rec]
        self.examples = self.examples[-200:]
        if self.example_path:
            self.example_path.parent.mkdir(parents=True, exist_ok=True)
            self.example_path.write_text("\n".join(json.dumps(e, ensure_ascii=False) for e in self.examples), encoding="utf-8")
        self._fit_examples()

    def similar_examples(self, question: str, k: int = 2, min_score: float = 0.25) -> list[dict]:
        if self._ex_vec is None:
            return []
        sims = linear_kernel(self._ex_vec.transform([question]), self._ex_mat).ravel()
        idx = [i for i in np.argsort(-sims)[:k] if sims[i] >= min_score]
        return [{**self.examples[i], "score": float(sims[i])} for i in idx]

    # ------------------------------------------------------------------ one call
    def retrieve(self, question: str, k_cat: int = 7, to_metric=None) -> Retrieval:
        r = Retrieval()
        sims = linear_kernel(self._cat_vec.transform([question]), self._cat_mat).ravel()
        for i in np.argsort(-sims)[:k_cat]:
            if sims[i] > 0.04:
                r.catalogue.append({"key": self._cat_keys[i], "text": self._cat_texts[i].split(" ", 2)[-1][:420]})
        ents = self.find_entities(question)
        if to_metric is not None:
            for e in ents:
                e["x"], e["y"] = to_metric(e["lon"], e["lat"])
        r.entities = ents if to_metric is not None else []
        r.examples = self.similar_examples(question)
        if self.kb is not None:
            try:
                r.docs = [{"cite": h["cite"], "text": h["text"]} for h in self.kb.search(question, 2)]
            except Exception:  # noqa: BLE001
                r.docs = []
        return r


def build_city_entities(partition_root: Path, registry: gpd.GeoDataFrame, metro_dir: Optional[Path] = None) -> pd.DataFrame:
    """Named places across the whole city from the stored layers (districts, neighbourhoods, attractions, metro stations, streets)."""
    rows = []
    for r in registry.itertuples():
        rows.append({"name": r.name, "name_ar": r.name_ar, "kind": f"{r.source} {r.kind}", "lon": r.centroid_lon, "lat": r.centroid_lat, "district": None})
    for layer, kind in (("attraction", "attraction"), ("street", "street")):
        d = Path(partition_root) / layer
        parts = []
        for p in d.glob("t*__*.parquet"):
            g = gpd.read_parquet(p, columns=["geometry", "name", "urbi_district_name"])
            g = g[g["name"].notna() & (g["name"].astype(str).str.len() > 2)]
            if layer == "street":
                g = g.drop_duplicates(subset=["name", "urbi_district_name"])
            c = g.geometry.representative_point()
            parts.append(pd.DataFrame({"name": g["name"].astype(str).to_numpy(), "name_ar": None, "kind": kind, "lon": c.x.to_numpy(), "lat": c.y.to_numpy(), "district": g["urbi_district_name"].to_numpy()}))
        if parts:
            rows += pd.concat(parts, ignore_index=True).drop_duplicates(subset=["name", "district"]).to_dict("records")
    if metro_dir is not None and (Path(metro_dir) / "stations.parquet").exists():
        s = gpd.read_parquet(Path(metro_dir) / "stations.parquet")
        for r in s.itertuples():
            rows.append({"name": None, "name_ar": r.name, "kind": "metro station", "lon": r.geometry.x, "lat": r.geometry.y, "district": None})
    return pd.DataFrame(rows)


def load_or_build_entities(cache: Path, partition_root: Path, registry: gpd.GeoDataFrame, metro_dir: Optional[Path] = None) -> pd.DataFrame:
    """The city-wide entity table is built once (about a minute) and cached as parquet."""
    cache = Path(cache)
    if cache.exists():
        return pd.read_parquet(cache)
    df = build_city_entities(partition_root, registry, metro_dir)
    df.to_parquet(cache)
    return df
