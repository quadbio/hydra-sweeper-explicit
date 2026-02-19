# Copilot Instructions for hydra-sweeper-explicit

## Project Overview

A Hydra sweeper plugin that runs **explicit parameter combinations** (no
Cartesian product). Each dict in `combinations` becomes one job; optionally
expanded across multiple seeds.

## Quick reference

| Task | Command |
|------|---------|
| Install | `uv sync` |
| Run tests | `hatch test` |

## Architecture

| File | Purpose |
|------|---------|
| `_sweeper.py` | `ExplicitSweeper(Sweeper)` — core logic, override formatting, seed expansion |
| `searchpath.py` | `SearchPathPlugin` — registers `conf/` with Hydra's config search path |
| `conf/` | Default Hydra config YAML for the sweeper |

## Hydra Plugin Conventions

- Package lives under `src/hydra_plugins/hydra_sweeper_explicit/` (Hydra's
  plugin discovery requires the `hydra_plugins` namespace package).
- `SearchPathPlugin` + `conf/__init__.py` make the bundled YAML discoverable.
- Users activate the sweeper via `hydra/sweeper=explicit` or by setting
  `_target_: hydra_sweeper_explicit.ExplicitSweeper` in their config.
