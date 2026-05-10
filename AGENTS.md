# AGENTS.md — hydra-sweeper-explicit

A [Hydra](https://hydra.cc) sweeper plugin that runs **explicit parameter combinations** (no Cartesian product), with optional seed expansion and per-combination launcher overrides for heterogeneous compute (e.g. mixed CPU/GPU SLURM partitions).

## Trust Order

When sources disagree:
1. PR description and changed code
2. This file (`AGENTS.md`)
3. `REVIEW_GUIDE.md`
4. Tests and fixtures
5. `README.md` (end-user docs)

Every fact should have one owner. This file owns invariants, development commands, and the reference table below.

## Where To Find What

| Topic | Source of truth |
|-------|----------------|
| End-user usage and installation | `README.md` |
| Plugin invariants and dev commands | This file |
| PR review workflow and risk areas | `REVIEW_GUIDE.md` |
| Core sweeper logic | `src/hydra_plugins/hydra_sweeper_explicit/_sweeper.py` |
| Hydra search-path registration | `src/hydra_plugins/hydra_sweeper_explicit/searchpath.py` |
| Bundled sweeper config | `src/hydra_plugins/hydra_sweeper_explicit/conf/hydra/sweeper/explicit.yaml` |
| Tests and fixtures | `tests/test_sweeper.py`, `tests/conftest.py` |

## Review Guidelines

For GitHub PR reviews, use `REVIEW_GUIDE.md` as the canonical review workflow and source of review-specific risk areas, testing checks, and documentation-impact checks. This file only owns the project invariants below.

## Critical Invariants

- **Plugin discovery.** The package must live under `src/hydra_plugins/hydra_sweeper_explicit/`. Hydra walks the `hydra_plugins` namespace package at startup; moving or renaming this path silently breaks plugin loading. The wheel ships the whole `src/hydra_plugins` tree (see `pyproject.toml` → `tool.hatch.build.targets.wheel.packages`).
- **Search-path mechanism.** `ExplicitSweeperSearchPathPlugin.manipulate_search_path()` appends `pkg://hydra_plugins.hydra_sweeper_explicit.conf` to Hydra's search path. There is no `conf/__init__.py` — package resource resolution does the work.
- **Activation.** Users either select the bundled config via `hydra/sweeper=explicit`, or set `_target_: hydra_plugins.hydra_sweeper_explicit.ExplicitSweeper` directly. The bare `hydra_sweeper_explicit` namespace is **not** importable.
- **Combination semantics.** `combinations` is exact — no Cartesian product. One dict in → one job (or N jobs if `seeds` is set). The seed axis is the only implicit expansion.
- **Reserved key `_launcher_`.** In each combination dict, `_launcher_` is filtered out before `_format_override` and must never appear in the override list passed to `launcher.launch()`.
- **Launcher grouping and concurrency.** Jobs are grouped by their `_launcher_` value; each group launches with its own launcher instance. Single-group sweeps stay single-threaded. Multi-group sweeps fan out via `ThreadPoolExecutor` so all submissions hit the queue together (launcher instances pre-built, `launch()` calls run in parallel).
- **Shared sweep dir.** `OmegaConf.copy_cache(from_config=self.config, to_config=new_config)` in `_make_launcher` propagates the resolved `${now:...}` value so every launcher group shares one timestamp and output directory. The `${now:...}` resolver is force-evaluated in `sweep()` before any group's `_make_launcher` call.
- **Output dir uniqueness.** `initial_job_idx` is incremented per group so output subdirectories stay distinct across launcher groups.
- **Index-aligned returns.** `sweep()`'s return list is aligned to input combination order, even though groups may complete out of order.
- **Override quoting.** `_format_override` lowercases bools, maps `None` → `null`, and double-quotes string values containing any of `" ,[]{}"`.
- **`launcher_config_group` semantics.** The default `hydra/launcher` routes the launcher override into Hydra-level overrides; setting it to `launcher` (used with `@package _global_` configs) routes into task-level overrides instead.

## Development Commands

Python 3.12, 3.13, and 3.14.

```bash
uv sync                           # install
hatch test                        # tests on the highest configured Python
hatch test --all                  # full matrix (3.12, 3.13, 3.13-pre)
pre-commit run --all-files        # ruff lint + format
```

Focused test runs:
```bash
uv run pytest tests/test_sweeper.py
uv run pytest tests/test_sweeper.py::TestSweep
```
