"""Provider interface shared by every source adapter."""
from __future__ import annotations

import abc
import datetime as dt
from dataclasses import dataclass, field
from typing import Any, Optional

import geopandas as gpd

BBox = tuple[float, float, float, float]  # west, south, east, north


@dataclass
class Capabilities:
    provider_id: str
    entity_types: list[str]
    attributes: dict[str, list[str]]  # entity type -> supported attributes
    geometry: dict[str, str]  # entity type -> geometry type
    crs: str = "EPSG:4326"
    notes: str = ""


@dataclass
class CoverageEstimate:
    provider_id: str
    entity_type: str
    bbox: BBox
    expected_features: Optional[int]  # None = unknown
    estimated_requests: int
    estimated_seconds: float
    basis: str  # how the estimate was made (never a made-up number)


@dataclass
class Provenance:
    provider_id: str
    entity_type: str
    dataset_version: str
    license: str
    attribution: str
    retrieved_at: str
    bbox: Optional[BBox]
    requests_made: int = 0
    from_cache: bool = False
    query_fingerprint: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return dict(self.__dict__)


@dataclass
class FeatureBatch:
    gdf: gpd.GeoDataFrame
    provenance: Provenance
    warnings: list[str] = field(default_factory=list)


def utcnow_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


class Provider(abc.ABC):
    provider_id: str = "abstract"

    @abc.abstractmethod
    def discover_capabilities(self) -> Capabilities: ...

    @abc.abstractmethod
    def estimate_coverage(self, entity_type: str, bbox: BBox) -> CoverageEstimate: ...

    @abc.abstractmethod
    def fetch_features(self, entity_type: str, bbox: BBox, *, budget=None, **kwargs) -> FeatureBatch: ...

    def fetch_details(self, entity_type: str, ids: list[str], *, budget=None) -> gpd.GeoDataFrame:
        """Second-stage enrichment by id. Providers without a detail endpoint return nothing."""
        return gpd.GeoDataFrame()

    @abc.abstractmethod
    def normalize(self, entity_type: str, raw) -> gpd.GeoDataFrame: ...

    @abc.abstractmethod
    def report_provenance(self) -> dict: ...
