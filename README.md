# GeoAI: geospatial acquisition and urban intelligence for Riyadh

Two connected Python projects.

| Project | What it is |
|---|---|
| [`geoacquirer/`](geoacquirer/README.md) | A geospatial data acquisition and verification engine: natural-language question, typed query, authorized sources, deterministic constraint evaluation, independent verification, evidence-backed exports. |
| [`geo-riyadh-ai/`](geo-riyadh-ai/README.md) | An urban-intelligence app built on it: a map of every district of Riyadh, plain-language answers, site suitability, facility-location optimisation, what-if scenarios, and an agent that writes and runs analysis code on the data. |

[`SOTA_references.md`](SOTA_references.md) lists the research the design draws on.

## Quick start

```bash
pip install -e ./geoacquirer -e "./geo-riyadh-ai[dev]"
cd geo-riyadh-ai
pytest                     # offline tests on synthetic data; no downloads, no keys needed
```

The tests and the synthetic demos need no data. The real app needs map data that is **not included** in this repository (some of it is licensed third-party data): see [`geo-riyadh-ai/docs/data_preparation.md`](geo-riyadh-ai/docs/data_preparation.md) and [`NOTICE.md`](NOTICE.md).

## Configuration

Copy `geo-riyadh-ai/.env.example` to `.env` and set a language-model key if you want the code-writing analysis agent (optional). Never commit `.env`. See [`SECURITY.md`](SECURITY.md).

## Status

Research-grade software, not a product. Results are estimates from open and licensed map data, not predictions of revenue or traffic; every analysis states its assumptions and limits. See each project's `docs/limitations.md`.

## Licence

Code: [MIT](LICENSE). Data: not included; see [`NOTICE.md`](NOTICE.md) for the terms of each source.
