# Agentic analysis: question -> plan -> code -> run -> observe -> plot -> explain

## Why
The fixed tools answer common questions well, but a real question ("residential houses with the best metro connection to the airport and parking for 10 vehicles within 1 km") needs a *new* analysis each time. The first version routed that question to the generic parking optimiser, which was wrong. The code agent writes and runs the analysis instead.

## Research basis (what each paper contributed here)
| Paper | Idea | Used as |
|---|---|---|
| ReAct, Yao et al. 2022 | interleave reasoning with actions and observations | the loop: plan, act, observe, revise |
| [PAL: Program-aided Language Models](https://arxiv.org/abs/2211.10435), [Program of Thoughts](https://arxiv.org/abs/2211.12588) | the model writes a program, an interpreter does the computing | all numbers come from executed code, never from the model's head |
| [CodeAct](https://arxiv.org/abs/2402.01030) | actions as executable Python (reported up to 20 % higher success than JSON tool calls) | the action is a Python script, not a tool-name + JSON |
| [Data Interpreter](https://arxiv.org/abs/2402.18679) | hierarchical plan, each sub-step refined and verified | explicit plan (interpretation, steps, assumptions) shown to the user |
| [LLM-Geo / Autonomous GIS](https://arxiv.org/abs/2305.06453) | question -> workflow of data operations -> code -> maps and charts | geospatial helpers and a map + chart per answer |
| [GeoAgent](https://arxiv.org/abs/2410.18792) | code interpreter + static analysis (+ RAG, MCTS) for geospatial code | static analysis before running; the schema is supplied instead of RAG; MCTS not implemented |
| Self-Debug (Chen et al., ICLR 2024), [revisiting self-debugging](https://arxiv.org/abs/2501.12793) | execution feedback repairs code, but weak feedback can mislead | bounded repair turns driven by real error text and automatic checks, not by self-written tests |
| [Self-consistency](https://arxiv.org/abs/2203.11171) | sample several solutions and keep the agreeing answer | **not implemented** (single solution); listed as the next verification step |
| [Recursive Language Models](https://arxiv.org/abs/2512.24601) | keep the large context outside the prompt, inspect it with code, optionally call sub-models | the data stays in the worker; the model sees only schemas and what its code prints. Recursive sub-model calls are **not implemented** |
| [DataFrame QA](https://arxiv.org/abs/2401.15463), [multimodal tabular agent](https://arxiv.org/abs/2511.18405) | write pandas from schema only; whitelisted libraries, no file/network, timeouts | the sandbox design |
| [DA-Code](https://arxiv.org/abs/2410.07331) | open-ended data-science agent tasks are hard (best model about 30 % on the benchmark) | why every answer shows its code, assumptions and checks instead of asking for blind trust |

Searches did not surface a paper on verifying generated-code answers against hallucination, so the verification here (below) is engineering practice, not a published method.

## How it works (`src/riyadhai/codeagent.py`)
1. **Data**: the layers of the selected area (plus a 1.5 km margin so edge distances are right) are written to a temporary folder. The prompt gets only schemas (columns, types, three example values), never rows.
2. **Plan + act**: the model (or a vetted recipe) returns JSON: interpretation, steps, assumptions and Python code that sets `result` and may draw a matplotlib figure.
3. **Run** (`sandbox.py`): static check (whitelisted imports; no open/exec/eval/getattr/dunder access; no file, network or system helpers), then a separate Python process with a cleaned environment (no API keys), a temp working directory and a wall-clock timeout. Not an OS sandbox: for untrusted multi-user hosting run the worker in a container.
4. **Observe**: the model sees the error (if any), what the code printed, the result's shape, columns, head and statistics, and automatic checks (empty result, all-NaN column, implausible units).
5. **Repair**: up to 5 turns; on error or a failed check the model gets the observation and rewrites the code.
6. **Verify**: automatic checks, plus a **number audit**: every number in the explanation must appear in what the code produced (as is, rounded, or as a percentage); otherwise it is shown as "check these numbers".
7. **Explain**: a plain-language headline, bullets, caveats and follow-ups, with the chart, the table (CSV/GeoJSON), the map layer, the code and the steps.

## What works today, and what does not
* Works and tested: sandbox safety (14 tests), the repair loop with a scripted stand-in model, number audit, recipe matching, end-to-end run of the vetted recipe through the API and UI.
* **Not tested with a real model**: no API key is configured in this environment (`/v1/health` reports `llm: false`). Set `ANTHROPIC_API_KEY` (optionally `RIYADHAI_LLM_MODEL`) or `RIYADHAI_LLM_URL` + `RIYADHAI_LLM_MODEL` (+ `RIYADHAI_LLM_KEY`) for any OpenAI-compatible endpoint. Until then only vetted recipes run; other custom questions are declined with that message, never guessed.
* Vetted recipe today: *homes near transit with parking* (`recipes.py`). More recipes are one function each.
* The quality of LLM-written code on this data has **not** been measured. Before relying on it, run a held-out question set and record pass rates, as done for the fixed tools.

## The metro-to-airport recipe: method and limits
* Metro lines and stations come from Suhail's vector tiles (6 lines, 92 stations; line lengths 14-45 km, close to the real network). The lines form one connected graph; ride distance = shortest path along the lines from a station to the nearest airport terminal station (terminals 1-2, 3-4, 5).
* Checked independently: ride distance is 1.2-3.0 times the straight-line distance (median 1.5), as expected for lines with transfers. Transfers are **not** penalised and no timetable exists, so minutes are estimates: walk = straight line x 1.3 at 5 km/h, 5 min wait, 35 km/h on the metro.
* Parking counts only lots with a reported capacity (unknown capacity is not counted), so true parking may be better than shown.
* In Al Murabba every residential building already meets the 10-vehicle rule, so the ranking depends only on the metro connection and the best homes differ by a few minutes.


## Added: geo-RAG, budgeted model calls (RLM step) and guarded data acquisition

### Geo-RAG (`georag.py`): retrieval finds, code computes
A retrieval layer feeds the agent the right slice of context; it never answers numeric questions itself (embeddings cannot do spatial predicates or aggregates).
* **Catalogue**: what each layer / column means, units, source, licence and pitfalls (for example `parking.capacity` is operator-supplied and often missing: unknown is not zero; `parcel.maxBuildingHeight` has no stated unit), plus live per-column stats of the current area.
* **Entities**: 64,993 named places (districts, neighbourhoods, attractions, metro stations, streets) in English and Arabic, matched by character n-gram similarity with a length-aware score, a small English-to-Arabic place-word table (so "Ministry of Finance" reaches the Arabic-only metro station) and a filter for generic names ("Park"). Resolved places are shown to the model with their match score and passed to the code as the dict `entities` (metric x/y).
* **Examples**: analyses that ran, passed the automatic checks and had no untraceable numbers are stored and retrieved as few-shot examples (`analyses/examples.jsonl`).
* **Reference notes**: the existing zoning / methods documents.
* Matching is lexical, not semantic: it handles spelling variants and the Arabic article, not synonyms ("skyscraper" will not find "Kingdom Centre"). Embeddings could be added later; their benefit has not been measured here.

### Budgeted model calls from code (`llmbridge.py`, the recursive step of RLM)
Inside the sandbox the code can call `llm_query(prompt)` or `llm_map(items, instruction)` for what code cannot do (classify or normalise fuzzy Arabic/English text, summarise). The sandbox still has no network and no keys: each call is a file handshake served by the parent process, which enforces a call budget (30), a per-call size limit (14,000 characters), a total size limit, depth 1 (a sub-call is a plain completion with no tools), a response cache and a full audit log shown in the technical details. Anything a sub-call returns is a model judgement; the final answer adds a caveat saying so. Only available when an LLM is configured. Not implemented: deeper recursion, parallel sub-calls.

### Guarded acquisition (`acquire.py`)
The agent can ask for missing data with typed actions `list_sources`, `estimate`, `fetch`; it cannot write download code. Guardrails: whitelisted layers only; Overture (open) is the only source the tool fetches from (Urbi and Suhail are reported as not usable here); area at most 0.1 deg2; at most 3 fetches per analysis; cached data needs no approval but a live download pauses the analysis (`needs_approval`) until the user agrees in the UI; every fetch is verified (rows, coverage of the box, empty or invalid geometry, missing key fields) and an empty result is reported as unknown, not zero. Credentials never reach the model or the sandbox.

### Status of these three additions
Tested with a scripted stand-in model: entity matching, catalogue retrieval, example memory, the bridge budget/cache/size limits, sandbox round trips, every acquisition guardrail, and an end-to-end run that resolves a named place, classifies text through `llm_map`, pauses for approval before a live download and resumes after approval (7 tests). **Not tested with a real model**: the quality of the model's plans, its use of `llm_query` and its choice of what to acquire are unmeasured; no key is configured in this environment.


## Live test with a real model (2026-10-20; `openai/gpt-oss-120b` on Groq, Al Olaya)
Run through the API with `mode=code`. Tests were small (one run each for most), so treat the results as a smoke test, not a benchmark.

| Test | Result |
|---|---|
| **Retrieval**: "How many residential buildings are within 500 metres of Kingdom Centre?" | Resolved the landmark from the place index, used `urbi_building.purpose_class`, answered **215** (22 s). Independent check from the stored data: 215 inside Al Olaya (280 including the part of the circle outside the district). Repeated twice: same answer. |
| **Keyword question**: pharmacies by name | 14, with example names; the model chose keyword matching (no model calls). |
| **Judgement question**: Indian-cuisine names | 6 after fixes; the model again chose keywords and made 0 model calls. |
| **Explicit `llm_map`**: classify 40 restaurant names by cuisine | Worked end to end: 4 sub-calls through the bridge, counts 2 / 13 / 2 / 5 / 18 (sum 40), answer caveated as model judgement; the model repaired its own `NameError` twice. |
| **Acquisition**: "How many road connectors ... use the connector layer" (layer not loaded) | First run: **wrong** (silently used the junction layer and called it connectors). After adding a deterministic gap detector to the prompt: it called `list_sources`, then `fetch`, and the agent paused for approval ("download connector from Overture ... about 3 minutes"). The approval was **not** granted in this test, so the downloaded-data path has not run with a real model. |

Problems the real model exposed, and what changed (all with tests):
* Provider token cap (HTTP 413 at 8,000 tokens per minute): prompt trimmed, older turns summarised, output limit computed from the prompt size, retries on 429/5xx.
* Empty replies from the reasoning model: one retry with a larger output limit, then a deterministic explanation built from the result itself (never a bare "finished").
* The code shown to the user was from a later failed attempt, not the one that produced the result: fixed.
* Entity lookups were case-sensitive and the value was not a geometry: now case-insensitive with `.x`, `.y`, `.geometry`.
* Silent layer substitution: a prompt rule was not enough for this model; the deterministic gap detector fixed it.
* The model prefers keyword lists even when asked to "judge" names; it uses `llm_map` when the user asks for it explicitly. Whether it should be pushed harder is a product decision (cost, latency, accuracy), not yet measured.
* The number audit cannot catch a semantic mistake (a correct-looking number computed from the wrong layer); the gap detector and the visible code/assumptions are the defences.

Model choice matters: `llama-3.1-8b-instant` (the model in `.env`) no longer exists on this Groq account (HTTP 404 `model_not_found`); available chat models were `openai/gpt-oss-120b`, `openai/gpt-oss-20b`, `qwen/qwen3.8-27b` and `allam-2-7b`. A small free-tier limit (8,000 tokens per minute) shaped the prompt-size work above. Not tried: `qwen/qwen3.8-27b`, `gpt-oss-20b`, a held-out question set, or Claude models.
