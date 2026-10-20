"""Bounded, resumable, cancellable crawler with backoff, jitter, checkpointing and dedup."""
from __future__ import annotations

import json
import random
import threading
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Hashable, Iterable, Optional

from ..security import BudgetExceeded, BudgetTracker, RateLimiter


class TransientError(Exception):
    """Retryable failure (timeouts, 5xx)."""


class RateLimitedError(TransientError):
    """HTTP 429: the crawler backs off harder and counts it toward the abort threshold."""


class FatalSourceError(Exception):
    """401/403/robots-style refusal: stop crawling this source; never try to evade."""


@dataclass
class CrawlStats:
    cells_total: int = 0
    cells_done: int = 0
    cells_failed: int = 0
    cells_skipped_checkpoint: int = 0
    retries: int = 0
    rate_limited: int = 0
    items: int = 0
    duplicates: int = 0
    aborted: Optional[str] = None
    errors: list = field(default_factory=list)


class Checkpoint:
    def __init__(self, path: Optional[Path]):
        self.path = Path(path) if path else None
        self.done: set[str] = set()
        self._lock = threading.Lock()
        if self.path and self.path.exists():
            self.done = set(json.loads(self.path.read_text(encoding="utf-8")).get("done", []))

    def mark(self, cell_id: str) -> None:
        with self._lock:
            self.done.add(cell_id)
            if self.path and len(self.done) % 1000 == 0:  # flush rewrites the whole file: keep it rare on city-scale crawls
                self.flush()

    def flush(self) -> None:
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(".tmp")
            tmp.write_text(json.dumps({"done": sorted(self.done)}), encoding="utf-8")
            tmp.replace(self.path)


class AdaptiveCrawler:
    """Runs `fetch(cell) -> list[item]` over cells.

    * concurrency is bounded by `max_workers` and the provider RateLimiter
    * transient errors retry with exponential backoff + full jitter (bounded by `max_retries`)
    * >`max_rate_limited` HTTP-429 events abort the crawl (we slow down or stop; we never evade)
    * `cancel` (threading.Event) stops promptly; the checkpoint makes the run resumable
    * items are de-duplicated by `key(item)`
    """

    def __init__(self, fetch: Callable[[Any], list], *, key: Callable[[Any], Hashable], cell_id: Callable[[Any], str] = str,
                 max_workers: int = 8, limiter: Optional[RateLimiter] = None, budget: Optional[BudgetTracker] = None,
                 max_retries: int = 4, base_delay: float = 0.5, max_delay: float = 20.0, max_rate_limited: int = 25,
                 checkpoint: Optional[Checkpoint] = None, cancel: Optional[threading.Event] = None, sleep=time.sleep,
                 on_items: Optional[Callable[[list], None]] = None):
        self.fetch, self.key, self.cell_id = fetch, key, cell_id
        self.max_workers, self.limiter, self.budget = max_workers, limiter or RateLimiter(0), budget
        self.max_retries, self.base_delay, self.max_delay = max_retries, base_delay, max_delay
        self.max_rate_limited = max_rate_limited
        self.checkpoint = checkpoint or Checkpoint(None)
        self.cancel = cancel or threading.Event()
        self.sleep = sleep
        self.on_items = on_items
        self.stats = CrawlStats()
        self.items: dict[Hashable, Any] = {}
        self._lock = threading.Lock()

    def _run_cell(self, cell) -> None:
        cid = self.cell_id(cell)
        for attempt in range(self.max_retries + 1):
            if self.cancel.is_set():
                return
            try:
                self.limiter.wait()
                if self.budget:
                    self.budget.charge_request()
                got = self.fetch(cell)
                new = []
                with self._lock:
                    for it in got:
                        k = self.key(it)
                        if k in self.items:
                            self.stats.duplicates += 1
                        else:
                            self.items[k] = it
                            new.append(it)
                    self.stats.items = len(self.items)
                if new and self.on_items:
                    self.on_items(new)
                if self.budget:
                    self.budget.charge_features(len(new))
                self.checkpoint.mark(cid)
                with self._lock:
                    self.stats.cells_done += 1
                return
            except FatalSourceError as e:
                self.stats.aborted = f"fatal source refusal: {e}"
                self.cancel.set()
                return
            except BudgetExceeded as e:
                self.stats.aborted = f"budget: {e}"
                self.cancel.set()
                return
            except TransientError as e:
                with self._lock:
                    self.stats.retries += 1
                    if isinstance(e, RateLimitedError):
                        self.stats.rate_limited += 1
                        if self.stats.rate_limited > self.max_rate_limited:
                            self.stats.aborted = "too many HTTP 429 responses; stopping instead of hammering the source"
                            self.cancel.set()
                            return
                delay = min(self.max_delay, self.base_delay * (2 ** attempt))
                if isinstance(e, RateLimitedError):
                    delay *= 2
                self.sleep(random.uniform(0, delay))  # full jitter
        with self._lock:
            self.stats.cells_failed += 1
            self.stats.errors.append(f"cell {cid}: retries exhausted")

    def run(self, cells: Iterable[Any]) -> CrawlStats:
        cells = list(cells)
        todo = []
        for c in cells:
            if self.cell_id(c) in self.checkpoint.done:
                self.stats.cells_skipped_checkpoint += 1
            else:
                todo.append(c)
        self.stats.cells_total = len(cells)
        pending = set()
        it = iter(todo)
        with ThreadPoolExecutor(self.max_workers) as ex:
            exhausted = False
            while (pending or not exhausted) and not self.cancel.is_set():
                while not exhausted and len(pending) < self.max_workers * 2:
                    try:
                        pending.add(ex.submit(self._run_cell, next(it)))
                    except StopIteration:
                        exhausted = True
                if not pending:
                    break
                done, pending = wait(pending, return_when=FIRST_COMPLETED, timeout=1.0)
            for f in pending:
                f.cancel()
        self.checkpoint.flush()
        if self.cancel.is_set() and not self.stats.aborted:
            self.stats.aborted = "cancelled"
        return self.stats
