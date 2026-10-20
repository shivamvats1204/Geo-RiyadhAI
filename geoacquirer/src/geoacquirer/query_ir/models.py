"""Typed Geographic Query Intermediate Representation (GeoQuery IR).

The IR is the *only* thing the deterministic engine executes. A language model (or the
rule-based compiler) may produce it, but it is validated strictly here, and no free-form
SQL or Python ever crosses this boundary.
"""
from __future__ import annotations

from enum import Enum
from typing import Annotated, Iterator, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class EntityType(str, Enum):
    POI = "poi"
    BUILDING = "building"
    PARCEL = "parcel"
    ROAD = "road"
    INTERSECTION = "intersection"
    PARKING = "parking"


class Op(str, Enum):
    WITHIN = "WITHIN"
    INTERSECTS = "INTERSECTS"
    TOUCHES = "TOUCHES"
    NEAR = "NEAR"
    FARTHER_THAN = "FARTHER_THAN"
    CONNECTED_TO = "CONNECTED_TO"
    AT_INTERSECTION = "AT_INTERSECTION"
    HAS_FRONTAGE_ON = "HAS_FRONTAGE_ON"
    INSIDE_BUILDING = "INSIDE_BUILDING"
    INSIDE_PARCEL = "INSIDE_PARCEL"
    NOT_WITHIN = "NOT_WITHIN"
    DISTANCE_BY_ROAD = "DISTANCE_BY_ROAD"
    CHANGED_BETWEEN = "CHANGED_BETWEEN"


class Exactness(str, Enum):
    EXACT = "exact"  # a deterministic geometric predicate over the stored geometry
    APPROXIMATE = "approximate"  # a documented approximation of the user's intent


# operations that need a numeric distance
_DISTANCE_OPS = {Op.NEAR, Op.FARTHER_THAN, Op.AT_INTERSECTION, Op.DISTANCE_BY_ROAD}
# operations that need an object to relate to
_OBJECT_OPS = {
    Op.WITHIN, Op.INTERSECTS, Op.TOUCHES, Op.NEAR, Op.FARTHER_THAN, Op.CONNECTED_TO,
    Op.HAS_FRONTAGE_ON, Op.INSIDE_BUILDING, Op.INSIDE_PARCEL, Op.NOT_WITHIN, Op.DISTANCE_BY_ROAD,
}


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ObjectSpec(_Strict):
    """What a predicate relates the subject to (a layer plus optional filters)."""

    entity_type: EntityType
    categories: list[str] = Field(default_factory=list)  # user-language or ontology ids
    road_classes: list[str] = Field(default_factory=list)  # e.g. ["motorway", "trunk"]
    alias: Optional[str] = None


class Predicate(_Strict):
    kind: Literal["predicate"] = "predicate"
    op: Op
    subject: str = "target"  # alias of an EntitySpec
    object: Optional[ObjectSpec] = None
    distance_m: Optional[float] = Field(default=None, gt=0, le=100_000)
    junction_kind: Optional[Literal["four_way", "t_junction", "any"]] = None
    min_count: int = Field(default=1, ge=1, le=100)  # e.g. frontage on >= 2 roads
    snapshots: Optional[tuple[str, str]] = None  # CHANGED_BETWEEN
    exactness: Exactness = Exactness.EXACT
    note: str = ""

    @model_validator(mode="after")
    def _check(self) -> "Predicate":
        if self.op in _DISTANCE_OPS and self.distance_m is None and self.op != Op.AT_INTERSECTION:
            raise ValueError(f"{self.op.value} requires distance_m")
        if self.op == Op.AT_INTERSECTION:
            if self.distance_m is None:
                raise ValueError("AT_INTERSECTION requires distance_m")
            if self.junction_kind is None:
                self.junction_kind = "four_way"
        if self.op in _OBJECT_OPS and self.object is None:
            raise ValueError(f"{self.op.value} requires an object")
        if self.op == Op.CHANGED_BETWEEN and not self.snapshots:
            raise ValueError("CHANGED_BETWEEN requires snapshots")
        return self


class And(_Strict):
    kind: Literal["and"] = "and"
    children: list["Expr"] = Field(min_length=1)


class Or(_Strict):
    kind: Literal["or"] = "or"
    children: list["Expr"] = Field(min_length=1)


class Not(_Strict):
    kind: Literal["not"] = "not"
    child: "Expr"


Expr = Annotated[Union[Predicate, And, Or, Not], Field(discriminator="kind")]
And.model_rebuild()
Or.model_rebuild()
Not.model_rebuild()


class EntitySpec(_Strict):
    alias: str = "target"
    entity_type: EntityType
    categories: list[str] = Field(default_factory=list)
    exclude_categories: list[str] = Field(default_factory=list)
    attributes: list[str] = Field(default_factory=list)


class AreaSpec(_Strict):
    name: str
    bbox: Optional[tuple[float, float, float, float]] = None  # (west, south, east, north)

    @field_validator("bbox")
    @classmethod
    def _bbox_ok(cls, v):
        if v is None:
            return v
        w, s, e, n = v
        if not (-180 <= w < e <= 180 and -90 <= s < n <= 90):
            raise ValueError("bbox must be (west, south, east, north) in WGS84 degrees")
        return v


class Budget(_Strict):
    max_requests: int = Field(default=200, ge=1, le=100_000)
    max_seconds: float = Field(default=900.0, gt=0, le=86_400)
    max_features: int = Field(default=2_000_000, ge=1)


class OutputSpec(_Strict):
    formats: list[Literal["csv", "geojson", "geoparquet", "jsonl", "map", "report"]] = Field(
        default_factory=lambda: ["csv", "geojson", "map"]
    )
    include_statuses: list[
        Literal["VERIFIED_MATCH", "PROBABLE_MATCH", "INSUFFICIENT_EVIDENCE", "REJECTED"]
    ] = Field(default_factory=lambda: ["VERIFIED_MATCH", "PROBABLE_MATCH"])
    requested_attributes: list[str] = Field(
        default_factory=lambda: ["name", "category", "coordinates", "building", "parcel", "nearby_roads"]
    )


class GeoQuery(_Strict):
    raw_text: str
    targets: list[EntitySpec] = Field(min_length=1)
    area: AreaSpec
    constraint: Optional[Expr] = None
    output: OutputSpec = Field(default_factory=OutputSpec)
    budget: Budget = Field(default_factory=Budget)
    freshness_days: Optional[int] = Field(default=None, ge=0)
    # defaults the compiler chose; always surfaced to the user
    defaults_applied: dict[str, Union[float, str, bool, list[str]]] = Field(default_factory=dict)
    ambiguities: list[str] = Field(default_factory=list)
    # spatial phrases the compiler could not turn into a predicate; they are NOT applied and are reported
    unparsed_phrases: list[str] = Field(default_factory=list)
    position_uncertainty_m: float = Field(default=10.0, ge=0, le=500)

    @model_validator(mode="after")
    def _aliases_unique(self) -> "GeoQuery":
        aliases = [t.alias for t in self.targets]
        if len(set(aliases)) != len(aliases):
            raise ValueError("target aliases must be unique")
        return self

    def primary(self) -> EntitySpec:
        return self.targets[0]


def iter_predicates(expr: Optional[Expr]) -> Iterator[Predicate]:
    if expr is None:
        return
    if isinstance(expr, Predicate):
        yield expr
    elif isinstance(expr, (And, Or)):
        for c in expr.children:
            yield from iter_predicates(c)
    elif isinstance(expr, Not):
        yield from iter_predicates(expr.child)
