"""Security primitives: allowlists, rate limits, budgets, safe identifiers, untrusted-text handling."""
from __future__ import annotations

import re
import threading
import time
from dataclasses import dataclass, field
from urllib.parse import urlparse


class SecurityError(RuntimeError):
    pass


class BudgetExceeded(SecurityError):
    pass


class ProviderDisabled(SecurityError):
    pass


class DomainNotAllowed(SecurityError):
    pass


ALLOWED_DOMAINS = {
    "overturemaps-us-west-2.s3.us-west-2.amazonaws.com",
    "overturemaps-us-west-2.s3.amazonaws.com",
    "s3.us-west-2.amazonaws.com",
    "stac.overturemaps.org",
    "overpass-api.de",
    "overpass.kumi.systems",
    "overpass.private.coffee",
    "nominatim.openstreetmap.org",
}


def check_domain(url_or_host: str, extra_allowed: set[str] | None = None) -> None:
    host = urlparse(url_or_host).hostname if "://" in url_or_host else url_or_host
    allowed = ALLOWED_DOMAINS | (extra_allowed or set())
    if host not in allowed:
        raise DomainNotAllowed(f"{host!r} is not on the source allowlist")


_IDENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_RELEASE = re.compile(r"^\d{4}-\d{2}-\d{2}\.\d+$")


def safe_identifier(name: str) -> str:
    if not _IDENT.match(name):
        raise SecurityError(f"unsafe SQL identifier: {name!r}")
    return name


def safe_release(name: str) -> str:
    if not _RELEASE.match(name):
        raise SecurityError(f"unsafe release string: {name!r}")
    return name


def sql_float(x: float) -> str:
    """Render a validated float for SQL (we never interpolate strings from users)."""
    v = float(x)
    if v != v or v in (float("inf"), float("-inf")):
        raise SecurityError("non-finite number")
    return repr(v)


@dataclass
class RateLimiter:
    """Thread-safe minimum-interval limiter (per provider)."""

    min_interval_s: float = 0.0
    _last: float = field(default=0.0, init=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False)

    def wait(self) -> None:
        if self.min_interval_s <= 0:
            return
        with self._lock:
            now = time.monotonic()
            delay = self._last + self.min_interval_s - now
            if delay > 0:
                time.sleep(delay)
            self._last = time.monotonic()


@dataclass
class BudgetTracker:
    max_requests: int
    max_seconds: float
    max_features: int
    requests: int = 0
    features: int = 0
    started: float = field(default_factory=time.monotonic)

    def charge_request(self, n: int = 1) -> None:
        self.requests += n
        self.check()

    def charge_features(self, n: int) -> None:
        self.features += n
        self.check()

    def check(self) -> None:
        if self.requests > self.max_requests:
            raise BudgetExceeded(f"request budget {self.max_requests} exceeded")
        if self.features > self.max_features:
            raise BudgetExceeded(f"feature budget {self.max_features} exceeded")
        if time.monotonic() - self.started > self.max_seconds:
            raise BudgetExceeded(f"time budget {self.max_seconds}s exceeded")

    def snapshot(self) -> dict:
        return {
            "requests": self.requests,
            "features": self.features,
            "elapsed_s": round(time.monotonic() - self.started, 2),
        }


_INJECTION_HINT = re.compile(
    r"(ignore (all |any )?(previous|prior) instructions|system prompt|you are now|act as|<\s*script)",
    re.I,
)


def sanitize_untrusted_text(text: str | None, max_len: int = 300) -> str | None:
    """Source-provided strings (names, addresses, notes) are data, never instructions.

    We strip control characters, bound the length and neutralize instruction-like phrases so the
    text is safe to show to a language model *as quoted data*. The agent never executes it.
    """
    if text is None:
        return None
    t = "".join(ch for ch in str(text) if ch.isprintable())[:max_len]
    if _INJECTION_HINT.search(t):
        t = "[flagged-untrusted] " + _INJECTION_HINT.sub("[removed]", t)
    return t
