# Data and third-party notice

**This repository contains source code only. It does not include any map data.** The code downloads or reads data you obtain yourself; you are responsible for complying with each source's terms.

| Source | Used for | Terms |
|---|---|---|
| [Overture Maps Foundation](https://overturemaps.org/) | places, roads, connectors, buildings, divisions (open) | Places: CDLA Permissive 2.0. Transportation / buildings: ODbL 1.0 (share-alike) plus per-record `sources.license`. Attribution required: (c) Overture Maps Foundation; (c) OpenStreetMap contributors where applicable. |
| [OpenStreetMap](https://www.openstreetmap.org/copyright) | street basemap (raster tiles), data inside Overture | ODbL 1.0. Attribution required. The public OSM tile server has a usage policy: do not use it for heavy or production traffic. |
| Urbi / 2GIS catalogue API | buildings, parking, streets, attractions, metro, administrative areas | **Not open data.** Requires your own API key and permission from the provider to store and use the data. The code ships with these providers **disabled** until you set an authorization reference (see `geoacquirer/docs/data_sources.md`). No Urbi data or key is included here. |
| Suhail | land parcels, zoning rules, neighbourhoods, metro lines | **Not open data.** Undocumented tile and API endpoints; use only with the provider's permission. Disabled by default in the same way. No Suhail data is included here. |

Do not redistribute data derived from Urbi or Suhail (tiles, exports, demo outputs) unless you hold the right to do so. The demo outputs under `examples/outputs` that were produced from real Suhail data are excluded from version control for this reason.

Language-model providers (Anthropic, OpenAI-compatible endpoints such as Groq or Ollama) are optional and configured with your own key in `.env`.
