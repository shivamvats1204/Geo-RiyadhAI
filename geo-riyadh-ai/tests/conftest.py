"""Test configuration. Tests must never depend on, or spend, a developer's real keys: .env loading is off and every model setting is cleared."""
import os

os.environ["RIYADHAI_NO_DOTENV"] = "1"
for _k in ("ANTHROPIC_API_KEY", "ANTHROPIC_BASE_URL", "RIYADHAI_LLM_URL", "RIYADHAI_LLM_MODEL", "RIYADHAI_LLM_KEY", "RIYADHAI_LLM_MAX_TOKENS",
           "GEOACQUIRER_LLM_URL", "GEOACQUIRER_LLM_MODEL", "GEOACQUIRER_LLM_KEY"):
    os.environ.pop(_k, None)

import pytest

from riyadhai.executor import Executor
from riyadhai.geoacquirer_client import GeoAcquirerClient
from riyadhai.rag import KnowledgeBase
from riyadhai.results import ResultStore
from riyadhai.testing import synthetic_area
from riyadhai.tools import ToolContext


@pytest.fixture(scope="session")
def world(tmp_path_factory):
    root = tmp_path_factory.mktemp("store")
    store, ds, truth = synthetic_area(root)
    out = root / "analyses" / "out"
    out.mkdir(parents=True, exist_ok=True)
    ctx = ToolContext(store, ds, ResultStore(store.root), KnowledgeBase(), GeoAcquirerClient(), out)
    return store, ds, truth, ctx


@pytest.fixture()
def executor(world):
    store, ds, truth, ctx = world
    ctx.approved = set()
    return Executor(ctx, store.root / "analyses" / "jobs")
