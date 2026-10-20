"""Category ontology: user language -> source taxonomy tokens.

Overture hierarchies look like ['shopping','food_and_beverage_store','grocery_store'], so a
concept matches when any of its tokens appears anywhere in a POI's category path. Other
sources (OSM tags, local CSVs) are mapped to the same token vocabulary by their adapters.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, Optional

import pandas as pd


@dataclass(frozen=True)
class Concept:
    id: str
    label: str
    tokens: frozenset
    exclude_tokens: frozenset = frozenset()
    synonyms_en: tuple = ()
    synonyms_ar: tuple = ()
    parent: Optional[str] = None
    entity_type: str = "poi"
    ambiguity: str = ""  # shown to the user when the term is genuinely ambiguous


def _c(id, label, tokens, en=(), ar=(), exclude=(), parent=None, entity_type="poi", ambiguity=""):
    return Concept(id, label, frozenset(tokens), frozenset(exclude), tuple(en), tuple(ar), parent, entity_type, ambiguity)


CONCEPTS: dict[str, Concept] = {c.id: c for c in [
    _c("shop", "Shops / retail stores", ["shopping"],
       en=("shop", "store", "retail", "retail store", "retail outlet", "boutique"),
       ar=("محل", "محلات", "متجر", "متاجر", "دكان"),
       ambiguity="'shops' mapped to the whole 'shopping' taxonomy branch; food & drink venues are excluded."),
    _c("supermarket", "Supermarkets / grocery", ["grocery_store", "supermarket", "convenience_store"],
       en=("supermarket", "grocery", "grocery store", "minimarket", "hypermarket"),
       ar=("سوبرماركت", "بقالة", "تموينات", "هايبر ماركت"), parent="shop"),
    _c("restaurant", "Restaurants", ["restaurant"], en=("restaurant", "diner"), ar=("مطعم", "مطاعم"),
       parent=None),
    _c("cafe", "Cafés", ["coffee_shop", "cafe", "coffee_and_tea"], en=("cafe", "coffee shop", "coffee"),
       ar=("مقهى", "كافيه")),
    _c("clinic", "Clinics", ["clinic", "doctor", "medical_center", "dental_clinic", "urgent_care_clinic"],
       en=("clinic", "clinics", "medical center", "doctor"), ar=("عيادة", "عيادات", "مركز طبي")),
    _c("healthcare", "Healthcare facilities", ["health_care"], exclude=["pharmacy"],
       en=("healthcare facility", "healthcare facilities", "healthcare", "health care", "medical facilities"),
       ar=("منشأة صحية", "رعاية صحية"),
       ambiguity="Healthcare facilities include clinics and hospitals but exclude pharmacies by default (configurable)."),
    _c("hospital", "Hospitals", ["hospital"], en=("hospital", "hospitals"), ar=("مستشفى", "مستشفيات")),
    _c("pharmacy", "Pharmacies", ["pharmacy"], en=("pharmacy", "pharmacies", "drugstore"), ar=("صيدلية", "صيدليات")),
    _c("school", "Schools", ["school", "elementary_school", "high_school", "primary_school", "secondary_school"],
       en=("school", "schools"), ar=("مدرسة", "مدارس")),
    _c("mosque", "Mosques", ["mosque", "place_of_worship"], en=("mosque", "mosques"), ar=("مسجد", "مساجد")),
    _c("bank", "Banks", ["bank", "banks_and_credit_unions", "atms"], en=("bank", "banks", "atm"), ar=("بنك", "بنوك")),
    _c("gas_station", "Fuel stations", ["gas_station", "fuel_station"], en=("gas station", "petrol station", "fuel station"),
       ar=("محطة وقود", "محطة بنزين")),
    _c("parking", "Parking facilities", ["parking", "parking_lot", "parking_garage"],
       en=("parking", "parking lot", "parking facility", "car park", "garage"), ar=("موقف", "مواقف", "موقف سيارات"),
       entity_type="parking"),
    _c("warehouse", "Warehouses / logistics", ["warehouse", "storage_facility", "logistics", "wholesale"],
       en=("warehouse", "warehouses", "storage", "depot"), ar=("مستودع", "مستودعات"),
       ambiguity="Warehouses are sparse in POI data; building class=warehouse/industrial is used when available.",
       entity_type="poi"),
]}


def _norm(t: str) -> str:
    t = t.lower().strip()
    t = re.sub(r"[^\w\s؀-ۿ]", " ", t)
    t = re.sub(r"\s+", " ", t)
    return t


def _stem(t: str) -> str:
    return re.sub(r"(ies)$", "y", re.sub(r"(es|s)$", "", t)) if len(t) > 3 and t.isascii() else t


def resolve(term: str) -> list[Concept]:
    """Longest-synonym match of a user term against the ontology."""
    n = _norm(term)
    hits: list[tuple[int, Concept]] = []
    for c in CONCEPTS.values():
        for syn in (*c.synonyms_en, *c.synonyms_ar, c.id.replace("_", " "), c.label.lower()):
            sn = _norm(syn)
            if sn and (n == sn or _stem(n) == _stem(sn) or re.search(rf"\b{re.escape(sn)}\b", n)):
                hits.append((len(sn), c))
                break
    hits.sort(key=lambda x: -x[0])
    seen, out = set(), []
    for _, c in hits:
        if c.id not in seen:
            seen.add(c.id)
            out.append(c)
    return out


def find_concepts_in_text(text: str) -> list[Concept]:
    n = _norm(text)
    found: list[tuple[int, int, Concept]] = []
    for c in CONCEPTS.values():
        best = None
        for syn in (*c.synonyms_en, *c.synonyms_ar):
            sn = _norm(syn)
            for m in re.finditer(rf"(?<!\w){re.escape(sn)}(s|es)?(?!\w)", n):
                if best is None or m.start() < best[0]:
                    best = (m.start(), len(sn))
        if best:
            found.append((best[0], -best[1], c))
    found.sort(key=lambda x: (x[0], x[1]))
    return [c for _, _, c in found]


def category_mask(path_series: pd.Series, concept_ids: Iterable[str], exclude_ids: Iterable[str] = ()) -> pd.Series:
    """Boolean mask over a Series of category-path lists."""
    inc: set = set()
    exc: set = set()
    for cid in concept_ids:
        c = CONCEPTS[cid]
        inc |= c.tokens
        exc |= c.exclude_tokens
    for cid in exclude_ids:
        exc |= CONCEPTS[cid].tokens

    def ok(path) -> bool:
        p = set(path) if path is not None and not (isinstance(path, float)) else set()
        if not p & inc:
            return False
        return not (p & exc)

    return path_series.apply(ok).astype(bool)  # an empty Series would otherwise stay object-dtype and index a frame as a column list


def shop_mask(path_series: pd.Series) -> pd.Series:
    """'Shops': the Overture 'shopping' branch (excludes food & drink venues by construction)."""
    return category_mask(path_series, ["shop"])
