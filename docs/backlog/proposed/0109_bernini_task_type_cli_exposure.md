# Proposed: Bernini task-type and guidance-mode CLI exposure

## Metadata

- Created: 2026-08-13
- Status: Proposed
- Completed: N/A

## ADR status

- Governing ADRs: [ADR 0006](../../adr/0006_generative_video_editing_task_boundary.md)
- ADR impact: none expected; this surfaces existing runtime parameters, not new task semantics.

## Context

The Bernini runtime accepts `task_type` and `guidance_mode` in `generate_video`
(`src/mflux/models/wan/variants/wan_bernini.py`), and the documented `mv2v` recipe — the one that
produces the full quadruped result on `v2v_case3` and is recipe-parity-proven against the
official implementation — is reachable only through the Python API. The CLI exposes
`--reference-guidance` and the other guidance scalars but not `--task-type` or
`--guidance-mode`, so `docs/bernini.md` documents the structure-changing recipe as
Python-API-only (Task-Specific Recipes section).

## Proposed work

1. Add `--task-type` (choices: the supported upstream task types) and optionally
   `--guidance-mode` to the Wan CLI for Bernini routes, validating combinations through the
   existing `ALLOWED_GUIDANCE_MODES_BY_TASK` table so invalid pairs fail closed with a clear
   message.
2. Record both values in metadata (already plumbed) and update `docs/bernini.md` so the
   structure-changing recipe has a CLI form.
3. Extend CLI tests for flag parsing, validation failures, and metadata capture.

## Validation expectations

- CLI run reproducing the `v2v_case3` mv2v-prefix result end-to-end.
- Tests cover accepted and rejected task/guidance combinations.
- Docs and `llms-full.txt` updated in the same change.

## References

- `docs/bernini.md` (Task-Specific Recipes)
- `src/mflux/models/wan/cli/wan_generate.py`
- Parity bundle rows `v2v_case3_mv2vprefix`, `r2v_case2_tuned`
