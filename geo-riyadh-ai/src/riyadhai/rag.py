"""Extractive geospatial RAG over methodology and data-definition documents.

Retrieval is TF-IDF over heading-delimited chunks. The platform returns the passages and their citation ids; it does not let a
model paraphrase numbers. Numeric geographic answers must come from deterministic tools (see tools.py).
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Optional

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel


_CAMEL = re.compile(r"([a-z0-9])([A-Z])")


def _split_camel(s: str) -> str:
    return _CAMEL.sub(r"\1 \2", s).lower()


@dataclass
class Chunk:
    doc: str
    heading: str
    text: str

    @property
    def cite(self) -> str:
        return f"[{self.doc}#{re.sub(r'[^a-z0-9]+', '-', self.heading.lower()).strip('-')}]"


def _chunks_from_md(doc: str, text: str) -> list[Chunk]:
    out, head, buf = [], "intro", []
    for line in text.splitlines():
        m = re.match(r"^(#{1,3})\s+(.*)", line)
        if m:
            if buf and "".join(buf).strip():
                out.append(Chunk(doc, head, "\n".join(buf).strip()))
            head, buf = m.group(2).strip(), []
        else:
            buf.append(line)
    if buf and "".join(buf).strip():
        out.append(Chunk(doc, head, "\n".join(buf).strip()))
    return out


class KnowledgeBase:
    def __init__(self, extra_dirs: Optional[list[Path]] = None):
        self.chunks: list[Chunk] = []
        pkg = resources.files("riyadhai.knowledge")
        for p in pkg.iterdir():
            if p.name.endswith(".md"):
                self.chunks += _chunks_from_md(p.name[:-3], p.read_text(encoding="utf-8"))
        for d in extra_dirs or []:
            for p in Path(d).glob("*.md"):
                self.chunks += _chunks_from_md(p.stem, p.read_text(encoding="utf-8"))
        # split camelCase field names (maxBuildingCoefficient -> max building coefficient) so natural-language queries match them
        self.vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), sublinear_tf=True,
                                   preprocessor=_split_camel)
        self.M = self.vec.fit_transform([f"{c.heading}. {c.text}" for c in self.chunks])

    def search(self, query: str, k: int = 3, min_score: float = 0.05) -> list[dict]:
        sims = linear_kernel(self.vec.transform([query]), self.M).ravel()
        order = sims.argsort()[::-1][:k]
        return [{"cite": self.chunks[i].cite, "doc": self.chunks[i].doc, "heading": self.chunks[i].heading, "score": round(float(sims[i]), 3),
                 "text": self.chunks[i].text} for i in order if sims[i] >= min_score]
