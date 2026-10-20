# Security notes

## Secrets
* API keys (language model, Urbi) belong in environment variables or a local `.env` file, which is git-ignored. Never put a key in source, tests, notebooks, logs or screenshots.
* If a key was ever typed into a chat, a notebook or a commit, treat it as exposed and rotate it.
* Run `git status` before every push and check that no `.env` file and no `data/` folder is staged. A secret scan in CI is a good addition (for example gitleaks).

## The code-writing analysis agent
LLM-written Python runs in a **separate process** with a static check (whitelisted imports; no file, network or system calls; no `eval`/`exec`/`getattr`/dunder access), a cleaned environment (no keys) and a time limit. This is **not an operating-system sandbox**. Do not expose the app to untrusted users unless the worker runs in a container (non-root, no network, no secrets) and you add authentication and rate limits.

## Running it for other people
The app keeps the selected area in shared state, so it is a **single-user tool today**. Before hosting it: per-session state, authentication, HTTPS, per-user model-cost limits, and a read-only mode that disables live downloads and the raw tool endpoint. Check the licence of every data source before serving derived tiles or exports publicly.

## Reporting a problem
Open an issue without including any key or data. For a vulnerability, contact the maintainers privately.
