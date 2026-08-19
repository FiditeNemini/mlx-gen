# Proposed: version label must prefer installed metadata over ancestor repo files

## Metadata

- Created: 2026-08-19
- Status: Proposed

## ADR status

- Governing ADRs: [ADR 0002](../../adr/0002_no_silent_automatic_fallbacks.md) (in spirit: a
  reported identity that does not match what is running is a silent substitution of truth).
- ADR impact: none; internal resolution-order fix.

## Context

`VersionUtil` resolves the CLI banner and release label by walking `Path(__file__).parents` for a
`pyproject.toml` / `CHANGELOG.md` before consulting installed distribution metadata. For a virtual
environment that lives inside the repository checkout - the default developer layout, and this
repository's own `.venv` - the walk escapes `site-packages`, reaches the repo root, and reports the
*checkout's* version regardless of what is actually installed.

## Problem

Observed during the 0.30.0 release audit: `.venv` contained a non-editable pip install of the
Aug-16 working tree (dist-info version 0.29.0, schema-8 capabilities, pre-contract SwiftVR CLI),
yet its `mlxgen` banner read `MLX-Gen 0.30.0 (2026-08-18)` because the parent walk found the repo's
bumped `pyproject.toml`. The stale install looked exactly like "the 0.30.0 release is broken". An
identity label that can disagree with the running code inverts its purpose.

## Proposed direction

1. When `mflux.__file__` resolves under a `site-packages`/`dist-packages` directory, trust
   `importlib.metadata` (version and, if needed, `PACKAGED_RELEASE_DATE`) and skip the parent walk
   entirely.
2. Keep the parent walk only for genuine from-source execution (`PYTHONPATH=src`, editable
   installs), where it is the correct source of truth.
3. Add a weight-free test that installs a mismatched stub dist-info alongside a repo-style tree and
   asserts the label follows the installed metadata.

## Non-goals

- No change to `PACKAGED_RELEASE_DATE` mechanics or the CHANGELOG date scan for source checkouts.

## Evidence

- 0.30.0 release audit (2026-08-19): `.venv/bin/python` label `0.30.0 (2026-08-18)` vs
  `importlib.metadata.version('mlx-gen')` returning `0.29.0` in the same interpreter.
