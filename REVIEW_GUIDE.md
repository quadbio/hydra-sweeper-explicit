# hydra-sweeper-explicit Review Guide

This file is the canonical, agent-neutral source of truth for automated PR review in this repo.
It is written for **agents performing PR reviews on GitHub** — use the imperative voice and be concrete.

**Scope: review only.** Your job is to produce review comments and suggestions on the PR. Do **not** push commits, modify files, or apply fixes yourself. Any changes are the author's call. Flag issues, ask questions, and suggest concrete diffs in comments when helpful — but leave the decision and the edits to the user.

Use `AGENTS.md` for architecture, invariants, and commands. Use this guide for review workflow, risk areas, testing checks, documentation-impact checks, and test lookup.

## Review-First Workflow

1. Read the PR body first when it is present.
2. Check CI status (`gh pr checks <num>`, `gh run view <run-id> --log-failed`) and investigate any test or lint failures before commenting.
3. Identify changed modules and map them to matching tests.
4. Check whether the change touches a repo invariant from `AGENTS.md`.
5. Prioritize behavioral regressions (override formatting, launcher dispatch, plugin discovery) over style feedback.
6. Verify that docs (human- and agent-facing) did not become stale — see [Documentation Impact](#documentation-impact).

## High-Risk Areas

Each bullet names the concern and points at the owning invariant(s) in `AGENTS.md`. Mechanics live there — the lines below only call out why these changes need a careful review.

- **Override formatting (`_format_override`).** See *Override quoting*. Subtle escaping changes silently break downstream Hydra parsing.
- **Launcher grouping and concurrent dispatch.** See *Launcher grouping and concurrency* and *Output dir uniqueness*. Regressions here are easy to miss because single-group sweeps still pass tests.
- **`_make_launcher` config re-composition.** See *Search-path mechanism*, *Shared sweep dir*, and *`launcher_config_group` semantics*. The hydra-vs-task override partitioning and the `copy_cache` propagation are both load-bearing for cross-group output dirs and timestamps.
- **Reserved-key handling.** See *Reserved key `_launcher_`*. Any change in the combination loop must preserve the filter.
- **Hydra plugin discovery.** See *Plugin discovery* and *Activation*. Path or packaging changes break loading silently — failures surface only at runtime.
- **Bundled config compatibility.** Keys in `conf/hydra/sweeper/explicit.yaml` are user-visible API. Renames are breaking; the `_target_` dotted path must stay consistent with what `__init__.py` re-exports (cross-checked against the *Activation* invariant).

## Changed-Path Test Lookup

Tests are concentrated in `tests/test_sweeper.py`. Mapping is small but non-symmetric — note the gaps.

| Changed file | Tests to check |
|--------------|---------------|
| `src/hydra_plugins/hydra_sweeper_explicit/_sweeper.py` | `tests/test_sweeper.py` (whole file) |
| `src/hydra_plugins/hydra_sweeper_explicit/__init__.py` | `tests/test_sweeper.py::TestExplicitSweeper::test_init_*`; verify `ExplicitSweeper` re-export path |
| `src/hydra_plugins/hydra_sweeper_explicit/searchpath.py` | No direct unit test today; flag a coverage gap when this file changes |
| `src/hydra_plugins/hydra_sweeper_explicit/conf/hydra/sweeper/explicit.yaml` | No automated check today; verify the YAML's `_target_` matches `__init__.py` re-exports and that all keys still match `ExplicitSweeper.__init__` parameters |

Cross-cutting fixture changes: inspect `tests/conftest.py`.

## Testing

Apply these checks whenever the PR touches code or tests.

**New code.** Confirm that new behavior is covered by tests.
- Reuse fixtures from `tests/conftest.py` rather than creating parallel ones.
- Prefer `pytest.mark.parametrize` over many near-identical tests.
- Favor few meaningful tests over many redundant ones; flag low-value tests that only duplicate existing coverage.

**Failing tests.** If CI is red, do not wave it through.
- Inspect which tests fail and why (`gh pr checks`, `gh run view --log-failed`).
- Distinguish critical regressions (override formatting, launcher dispatch, plugin discovery) from trivial or flaky failures.
- Surface critical failures back to the author and ask them to fix before merge.

**Modified tests.** Scrutinize *how* existing tests were changed.
- PRs that only relax thresholds, remove assertions, delete cases, or loosen `parametrize` matrices are a red flag — tests-working-around-tests defeats the purpose.
- Require an explicit justification in the PR body for any weakened assertion; do not accept silently.

## Documentation Impact

A single behavioral or API change often touches docs in multiple places. Point to the **owning file** for each topic rather than duplicating content in your review.

- API signature, public symbol, or invariant changes → `AGENTS.md` (Critical Invariants).
- Development command changes → `AGENTS.md` (Development Commands).
- End-user usage or installation changes → `README.md`.
- Review workflow, risk areas, or testing conventions changed → this file.
- Repo structure, new top-level docs, or moved pointers → `AGENTS.md` "Where To Find What" table, `CLAUDE.md`, `.github/copilot-instructions.md`.

If behavior changes but the relevant docs do not, call it out explicitly in the review and request the update.

## Review Checklist

- Does the change preserve the invariants in `AGENTS.md`?
- Does CI pass, and were any failures investigated? (See [Testing](#testing).)
- Is test coverage adequate and non-redundant, and are modified tests not simply weakened? (See [Testing](#testing).)
- Does it alter override formatting, launcher dispatch, or plugin discovery in a way that needs explicit release-note-style mention in the PR?
- Does it change the bundled config keys or the `_target_` dotted path (user-visible API)?
- Are all affected human- and agent-facing docs updated? (See [Documentation Impact](#documentation-impact).)
- Is the PR scope tight — no unrelated changes bundled in?

## PR Metadata

This repo uses a structured PR template at `.github/PULL_REQUEST_TEMPLATE.md`.

Reviewers and agents should treat these sections as the preferred summary surface:
- summary
- behavior or invariants changed
- tests run
- reviewer focus
- context
- open questions or follow-ups
