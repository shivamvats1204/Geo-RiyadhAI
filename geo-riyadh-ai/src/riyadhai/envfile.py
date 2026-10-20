"""Tiny `.env` loader (no extra dependency). Variables already set in the real environment always win; a missing file is fine.

Searched in order: the current directory, then the project folder (the one containing `src/`). Lines look like KEY=value; `#` starts a comment; values may be quoted.
The file is read once at startup by the server (and the CLI); keep it out of version control (it is listed in .gitignore).
"""
from __future__ import annotations

import os
from pathlib import Path


def parse_env_text(text: str) -> dict:
    out = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.lower().startswith("export "):
            line = line[7:].strip()
        k, v = line.split("=", 1)
        k, v = k.strip(), v.strip()
        if len(v) >= 2 and v[0] == v[-1] and v[0] in "\"'":
            v = v[1:-1]
        elif " #" in v:
            v = v.split(" #", 1)[0].rstrip()
        if k:
            out[k] = v
    return out


def load_dotenv(paths=None) -> list[str]:
    """Load .env files into os.environ without overriding existing variables; returns the files that were read."""
    if os.environ.get("RIYADHAI_NO_DOTENV") == "1":   # tests and CI: never pick up a developer's real keys
        return []
    project = Path(__file__).resolve().parents[2]
    candidates = [Path(p) for p in paths] if paths else [Path.cwd() / ".env", project / ".env"]
    read = []
    for f in candidates:
        if f.is_file() and str(f) not in read:
            for k, v in parse_env_text(f.read_text(encoding="utf-8-sig")).items():
                os.environ.setdefault(k, v)
            read.append(str(f))
    return read
