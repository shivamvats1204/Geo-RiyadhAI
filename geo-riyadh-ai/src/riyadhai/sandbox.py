"""Run LLM-written analysis code on the data, safely enough for a trusted-user tool, and report what happened.

Layers of protection (honest scope: this is NOT an OS-level sandbox; for untrusted multi-tenant use, run the worker inside a container/VM):
  1. static check (AST): only whitelisted imports; no open/exec/eval/__import__/getattr/...; no dunder attribute access; no file or network helpers
     (read_*/to_csv/to_parquet/savefig/...); no `while True`-style infinite loops without a break is NOT detected: the timeout handles it.
  2. separate Python process (`python -E`: environment variables ignored), clean environment (no API keys, no proxies), temp working directory, wall-clock timeout (killed on expiry).
  3. restricted builtins and an import hook inside the worker that re-checks every import at run time.
The worker receives the (scope-clipped) layers as parquet files, so the model never needs the data in its prompt: it only sees schemas and what its code prints.
"""
from __future__ import annotations

import ast
import json
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

ALLOWED_IMPORTS = {"pandas", "numpy", "geopandas", "shapely", "matplotlib", "scipy", "sklearn", "math", "statistics", "json", "re", "collections", "itertools",
                   "datetime", "networkx", "functools", "heapq", "pyproj", "seaborn", "textwrap", "operator", "bisect", "random"}
BLOCKED_NAMES = {"open", "exec", "eval", "compile", "__import__", "input", "breakpoint", "globals", "locals", "vars", "getattr", "setattr", "delattr", "memoryview", "exit", "quit",
                 "help", "dir"}
BLOCKED_ATTRS = {"read_csv", "read_parquet", "read_file", "read_excel", "read_json", "read_pickle", "read_sql", "read_html", "read_feather", "read_table", "read_xml",
                 "to_csv", "to_parquet", "to_file", "to_pickle", "to_excel", "to_sql", "to_feather", "to_html", "to_xml", "to_hdf", "savefig", "save", "load", "loadtxt", "savetxt",
                 "system", "popen", "remove", "rmdir", "unlink", "mkdir", "makedirs", "rename", "urlopen", "urlretrieve", "run", "call", "Popen", "fromfile", "tofile",
                 "environ", "getenv", "putenv", "listdir", "walk", "scandir", "glob", "write_text", "read_text", "write_bytes", "read_bytes"}
MAX_CODE_CHARS = 12000


@dataclass
class RunResult:
    ok: bool
    error: str = ""
    stdout: str = ""
    result_kind: str = "none"        # geodataframe | dataframe | json | none
    result_path: str = ""
    figure_paths: list = field(default_factory=list)
    notes: list = field(default_factory=list)
    seconds: float = 0.0
    workdir: str = ""


def validate_code(code: str) -> list[str]:
    """Static checks; returns a list of problems (empty = acceptable)."""
    problems: list[str] = []
    if len(code) > MAX_CODE_CHARS:
        return [f"code is too long ({len(code)} > {MAX_CODE_CHARS} characters); write a shorter solution"]
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return [f"syntax error: {e.msg} (line {e.lineno})"]
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                if a.name.split(".")[0] not in ALLOWED_IMPORTS:
                    problems.append(f"import of '{a.name}' is not allowed (allowed: {', '.join(sorted(ALLOWED_IMPORTS))})")
        elif isinstance(node, ast.ImportFrom):
            if (node.module or "").split(".")[0] not in ALLOWED_IMPORTS or node.level:
                problems.append(f"import from '{node.module}' is not allowed")
        elif isinstance(node, ast.Name) and node.id in BLOCKED_NAMES:
            problems.append(f"use of '{node.id}' is not allowed")
        elif isinstance(node, ast.Attribute):
            if node.attr.startswith("__"):
                problems.append(f"access to '{node.attr}' is not allowed")
            elif node.attr in BLOCKED_ATTRS:
                problems.append(f"'.{node.attr}' is not allowed (no file, network or system access)")
        elif isinstance(node, (ast.Global, ast.Nonlocal)):
            problems.append("global/nonlocal statements are not allowed")
    return sorted(set(problems))


def run_code(code: str, layers_dir: Path, meta: dict, timeout_s: float = 90.0, bridge=None) -> RunResult:
    """Execute `code` in a fresh process. `layers_dir` holds <name>.parquet; `meta` is JSON-able (crs, scope wkt, entities).

    `bridge(prompt, opts) -> str`, if given, services the code's llm_query / llm_map calls (file handshake; the worker has no network and no keys).
    Time spent inside the bridge does not count against the code's own timeout."""
    import time
    problems = validate_code(code)
    if problems:
        return RunResult(False, "code rejected before running: " + "; ".join(problems))
    work = Path(tempfile.mkdtemp(prefix="riyadhai_run_"))
    (work / "code.py").write_text(code, encoding="utf-8")
    (work / "rpc").mkdir()
    (work / "meta.json").write_text(json.dumps({**meta, "layers_dir": str(layers_dir), "allowed": sorted(ALLOWED_IMPORTS), "llm_enabled": bridge is not None}), encoding="utf-8")
    runner = work / "run.py"  # a copy in the temp folder: sys.path[0] is then that folder, so nothing from the app package can shadow a module
    shutil.copy(Path(__file__).with_name("sandbox_runner.py"), runner)
    env = {k: os.environ[k] for k in ("SYSTEMROOT", "PATH", "TEMP", "TMP", "USERPROFILE", "APPDATA", "LOCALAPPDATA", "HOME", "PYTHONUTF8") if k in os.environ}
    env["PYTHONUTF8"] = "1"
    env["MPLBACKEND"] = "Agg"
    t0 = time.monotonic()
    deadline = t0 + timeout_s
    done: set = set()
    with open(work / "stdout.txt", "w") as so, open(work / "stderr.txt", "w") as se:
        proc = subprocess.Popen([sys.executable, "-E", "-X", "utf8", str(runner), str(work)], cwd=work, env=env, stdout=so, stderr=se)
        while proc.poll() is None:
            for f in sorted((work / "rpc").glob("req_*.json")):
                if f.name in done:
                    continue
                done.add(f.name)
                t1 = time.monotonic()
                try:
                    req = json.loads(f.read_text(encoding="utf-8"))
                    resp = {"ok": True, "text": bridge(req["prompt"], req.get("opts", {}))} if bridge else {"ok": False, "error": "no language model is configured for llm_query"}
                except Exception as e:  # noqa: BLE001  (budget exceeded, model error...): the code sees it as an exception it can handle
                    resp = {"ok": False, "error": f"{type(e).__name__}: {e}"[:400]}
                out = work / "rpc" / f.name.replace("req_", "resp_")
                tmp = out.with_suffix(".tmp")
                tmp.write_text(json.dumps(resp), encoding="utf-8")
                os.replace(tmp, out)
                deadline += time.monotonic() - t1
            if time.monotonic() > deadline:
                proc.kill()
                proc.wait()
                return RunResult(False, f"the code ran longer than {timeout_s:.0f} s and was stopped; use a simpler or vectorised approach", seconds=timeout_s, workdir=str(work))
            time.sleep(0.05)
    sec = time.monotonic() - t0
    st_path = work / "status.json"
    if not st_path.exists():
        return RunResult(False, "the worker crashed: " + (work / "stderr.txt").read_text(errors="replace")[-600:], seconds=sec, workdir=str(work))
    st = json.loads(st_path.read_text(encoding="utf-8"))
    return RunResult(bool(st["ok"]), st.get("error", ""), st.get("stdout", ""), st.get("result_kind", "none"), st.get("result_path", ""), st.get("figures", []), st.get("notes", []), sec, str(work))


def cleanup(res: RunResult) -> None:
    if res.workdir:
        shutil.rmtree(res.workdir, ignore_errors=True)
