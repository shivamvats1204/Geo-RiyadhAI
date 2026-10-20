"""Append-only JSONL audit log (who asked what, which sources were touched, what was exported)."""
from __future__ import annotations

import datetime as dt
import json
import threading
from pathlib import Path

from ..security import sanitize_untrusted_text


class AuditLog:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def event(self, kind: str, **fields) -> None:
        rec = {"ts": dt.datetime.now(dt.timezone.utc).isoformat(timespec="milliseconds"), "event": kind}
        rec.update({k: (sanitize_untrusted_text(v, 500) if isinstance(v, str) else v) for k, v in fields.items()})
        with self._lock, open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
