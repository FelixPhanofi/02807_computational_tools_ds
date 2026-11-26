> Guidance for autonomous coding agents. Read this before writing, editing, or executing anything in this repo.

# Commands

* `uv` manages the environment: `uv add <pkg>` to install, `uv run <cmd>` for everything Python. `uv sync` (core), `--extra notebook` (Jupyter, matplotlib), `--extra gpu` (CuPy, CUDA 12). One lockfile covers all.
* Tests: `uv run pytest tests/` (no data or model needed; GPU tests skip without CuPy). Lint/format: `uv run ruff check . --fix`, `uv run ruff format .` (or `uv run invoke fmt`). Tasks: `uv run invoke --list` (`test`, `data`, `fmt`; see `tasks.py`). Hooks: `uv run pre-commit run --all-files` (config: `.pre-commit-config.yaml`).
* Do not run these unless asked (large downloads or long GPU jobs, all resumable):
  * `uv run python -m afp.data --subset small|medium [--limit N]` (or `uv run invoke data`): downloads about 8 GB (`small`) or 22 GB (`medium`).
  * `uv run python -m afp.fingerprint cache|train|encode` (see `README.md`): `cache` downloads MuQ weights (about 1.2 GB) and runs the model over every window (about 35,000 clean windows for `small`, plus pairs and queries).
* Cheap CPU commands on existing fingerprints: `uv run python -m afp.cluster [--k ...] [--nprobe ...]` (k-means IVF evaluation), `uv run python -m afp.related [--min-similarity 0.9]` (DBSCAN, tracks sharing audio).
* Notebooks and the medium rebuild are in `README.md` (LSH section).

# Layout

* Source in `src/afp/` (`afp` is a placeholder name); module map in `README.md`.
* Tests must not need audio data or a downloaded model.
* Audio, cached embeddings and fingerprints live under `data/` and are git-ignored.

# Code style and documentation

* Follow existing style; line length 120; f-strings; type hints; no inline comments unless absolutely necessary.
* All functions and classes have Google-style docstrings.
* Update this file when tools or commands are added.

The report replay and the three optional full measurement commands are documented in [reports/report-evidence/README.md](reports/report-evidence/README.md).
The Signal Index explorer uses explorer as its Vercel root. Its local start/stop and evidence rebuild commands are in explorer/README.md.
