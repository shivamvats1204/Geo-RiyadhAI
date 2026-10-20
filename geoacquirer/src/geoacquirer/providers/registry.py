"""Machine-readable source registry + deterministic source scoring."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from importlib import resources
from typing import Optional

from ..security import ProviderDisabled


def load_registry() -> dict:
    return json.loads(resources.files("geoacquirer.providers").joinpath("registry.json").read_text(encoding="utf-8"))


def is_authorized(provider_id: str, entry: dict) -> tuple[bool, str]:
    """Providers flagged enabled=false need an explicit, recorded authorization reference."""
    if entry.get("enabled", False):
        return True, "open"
    p = provider_id.upper()
    if os.environ.get(f"GEOACQUIRER_ENABLE_{p}") == "1" and os.environ.get(f"GEOACQUIRER_{p}_AUTHORIZATION_REF"):
        return True, os.environ[f"GEOACQUIRER_{p}_AUTHORIZATION_REF"]
    return False, "no recorded authorization (set GEOACQUIRER_ENABLE_%s=1 and GEOACQUIRER_%s_AUTHORIZATION_REF)" % (p, p)


def require_authorized(provider_id: str) -> str:
    ok, ref = is_authorized(provider_id, load_registry()[provider_id])
    if not ok:
        raise ProviderDisabled(f"{provider_id} is disabled: {ref}")
    return ref


@dataclass
class SourceScore:
    provider_id: str
    score: float
    parts: dict
    eligible: bool
    reason: str = ""


# documented weights (see docs/planning.md); higher score = preferred
W = {"coverage": 0.35, "reliability": 0.25, "freshness": 0.15, "cost": 0.15, "local": 0.10}
_FRESH = {"minutes": 1.0, "monthly": 0.8, "unknown": 0.3}


def score_sources(entity_type: str, local_provider_ids: set[str], registry: Optional[dict] = None,
                  observed: Optional[dict] = None) -> list[SourceScore]:
    reg = registry or load_registry()
    observed = observed or {}
    out = []
    for pid, e in reg.items():
        ok, why = is_authorized(pid, e)
        if entity_type not in e["entity_types"]:
            continue
        if not ok:
            out.append(SourceScore(pid, 0.0, {}, False, why))
            continue
        cov = 1.0 if e.get("coverage") == "global" else 0.6
        rel = observed.get(pid, {}).get("success_rate", e["reliability"].get("prior_success_rate") or 0.5)
        fr = e.get("freshness", "")
        fresh = _FRESH["minutes"] if "minutes" in fr else _FRESH["monthly"] if "monthly" in fr else _FRESH["unknown"]
        sec = e["estimated_cost"]["seconds_per_layer"]
        cost = 1.0 / (1.0 + sec / 120.0)
        local = 1.0 if pid in local_provider_ids else 0.0
        parts = {"coverage": cov, "reliability": rel, "freshness": fresh, "cost": cost, "local": local}
        score = sum(W[k] * v for k, v in parts.items())
        out.append(SourceScore(pid, round(score, 4), parts, True))
    return sorted(out, key=lambda s: (-s.eligible, -s.score))
