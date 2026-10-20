"""DataContext: the metric-CRS layers a query is executed against."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import geopandas as gpd
from pyproj import CRS

from ..spatial import JunctionConfig, build_junctions, find_grade_separated_crossings, prepare_segments
from ..spatial.network import RoadNetwork


@dataclass
class DataContext:
    crs: CRS
    layers: dict[str, gpd.GeoDataFrame] = field(default_factory=dict)
    connectors: Optional[gpd.GeoDataFrame] = None
    provenance: dict[str, dict] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    snapshots: dict[str, gpd.GeoDataFrame] = field(default_factory=dict)  # label -> POI snapshot
    junction_cfg: JunctionConfig = field(default_factory=JunctionConfig)
    area: Optional[object] = None  # shapely polygon (metric CRS) restricting *subjects*; objects may lie outside
    _network: Optional[RoadNetwork] = None

    def has(self, name: str) -> bool:
        g = self.layers.get(name)
        return g is not None

    def add_roads(self, segments: gpd.GeoDataFrame, connectors: gpd.GeoDataFrame) -> None:
        seg = prepare_segments(segments, self.junction_cfg)
        self.layers["road"] = seg
        self.connectors = connectors
        j = build_junctions(segments, connectors, self.junction_cfg)
        self.layers["junction"] = j
        self.layers["excluded_crossings"] = find_grade_separated_crossings(segments)
        self.notes.append(
            f"{len(seg)} drivable segments, {len(j)} junction clusters "
            f"({int((j.kind == 'four_way').sum())} four-way, {int((j.kind == 't_junction').sum())} T), "
            f"{len(self.layers['excluded_crossings'])} grade-separated crossings excluded"
        )

    def network(self) -> RoadNetwork:
        if self._network is None:
            if self.connectors is None or not self.has("road"):
                raise RuntimeError("road layer not loaded")
            self._network = RoadNetwork(self.layers["road"], self.connectors)
        return self._network
