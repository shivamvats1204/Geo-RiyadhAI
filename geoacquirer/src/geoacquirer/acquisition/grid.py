"""Geographic partitioning strategies for crawling providers that answer per-cell requests."""
from __future__ import annotations

import math
from typing import Callable, Iterable

BBox = tuple[float, float, float, float]  # west, south, east, north


def uniform_grid(bbox: BBox, step_deg: float) -> list[BBox]:
    w, s, e, n = bbox
    nx = max(1, math.ceil((e - w) / step_deg))
    ny = max(1, math.ceil((n - s) / step_deg))
    cells = []
    for i in range(nx):
        for j in range(ny):
            cells.append((w + i * step_deg, s + j * step_deg, min(e, w + (i + 1) * step_deg), min(n, s + (j + 1) * step_deg)))
    return cells


def partition(bbox: BBox, max_cells: int) -> list[BBox]:
    """Smallest uniform grid with at most `max_cells` cells (bounding-box partitioning)."""
    w, s, e, n = bbox
    k = max(1, int(math.floor(math.sqrt(max_cells))))
    step = max((e - w) / k, (n - s) / k)
    return uniform_grid(bbox, step)


def quadtree(bbox: BBox, count_fn: Callable[[BBox], int], max_per_cell: int, max_depth: int = 6, min_size_deg: float = 0.0005) -> list[tuple[BBox, int]]:
    """Subdivide cells whose feature count exceeds `max_per_cell` (density-aware).

    `count_fn(cell)` is the (cheap) probe, e.g. a provider's count endpoint or a coarse scout request.
    Returns leaf cells with their counts; empty cells are dropped.
    """
    out: list[tuple[BBox, int]] = []

    def rec(cell: BBox, depth: int):
        c = count_fn(cell)
        if c == 0:
            return
        w, s, e, n = cell
        if c <= max_per_cell or depth >= max_depth or (e - w) <= min_size_deg or (n - s) <= min_size_deg:
            out.append((cell, c))
            return
        mx, my = (w + e) / 2, (s + n) / 2
        for sub in ((w, s, mx, my), (mx, s, e, my), (w, my, mx, n), (mx, my, e, n)):
            rec(sub, depth + 1)

    rec(bbox, 0)
    return out


def coarse_to_fine(bbox: BBox, coarse_step: float, fine_step: float, probe: Callable[[BBox], int]) -> list[BBox]:
    """Scout on a coarse grid, then return fine cells only inside non-empty coarse cells (as in urbi.ipynb)."""
    fine: list[BBox] = []
    for cell in uniform_grid(bbox, coarse_step):
        if probe(cell) > 0:
            fine.extend(uniform_grid(cell, fine_step))
    return fine


def prioritize_by_density(cells: Iterable[BBox], density: Callable[[BBox], float]) -> list[BBox]:
    return sorted(cells, key=lambda c: -density(c))


def tile_xy(lon: float, lat: float, z: int) -> tuple[int, int]:
    n = 2 ** z
    x = int((lon + 180.0) / 360.0 * n)
    lat_rad = math.radians(lat)
    y = int((1 - math.log(math.tan(lat_rad) + 1 / math.cos(lat_rad)) / math.pi) / 2 * n)
    return x, y


def tiles_for_bbox(bbox: BBox, z: int) -> list[tuple[int, int]]:
    w, s, e, n = bbox
    x0, y0 = tile_xy(w, n, z)
    x1, y1 = tile_xy(e, s, z)
    return [(x, y) for x in range(min(x0, x1), max(x0, x1) + 1) for y in range(min(y0, y1), max(y0, y1) + 1)]
