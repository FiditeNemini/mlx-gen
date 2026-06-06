# Completed: Release validation registry for I2I evidence

## Metadata
- Created: 2026-06-05
- Completed: 2026-06-05
- Status: Completed

## ADR status
- Governing ADRs: [ADR 0001](../../adr/0001_runtime_smoke_validation_for_model_routes.md), [ADR 0002](../../adr/0002_no_silent_automatic_fallbacks.md)
- ADR impact: None

## Context

The I2I edit matrices showed two different concepts that were too easy to confuse:

- route capability: MLX-Gen can dispatch and validate a request shape;
- release validation: an exact source or prepared model package passed, partially passed, failed,
  or became stale under a named evidence profile.

This distinction matters for AbstractVision and other Python callers. They need stable routing APIs
without hiding explicit user choices, but they also need a machine-readable way to show warnings or
filter release-quality model/package rows.

## Current code reality

- `src/mflux/task_inference.py` exposes route capabilities and generation plans.
- `mlxgen capabilities` prints that route contract.
- The 2026-06-05 I2I edit proof assets are in
  `validation_outputs/edit_prepared_capability_2026_06_05/`.
- FIBO Edit is unavailable through unified public capability discovery and remains a failing
  validation row.
- Qwen Image Edit 2509 and 2511 have different source/q8/q4 validation profiles despite sharing
  similar route modes.

## Problem

`mlxgen capabilities` could be misread as “this model/package passed visual QA.” That is wrong for
FIBO Edit and for Qwen Image Edit 2511 multi-reference composition. The report artifacts also needed
an authoritative, structured status source with reviewer notes instead of hard-coded labels in a
contact-sheet script.

## What changed

- Added `src/mflux/release/validation_registry.py` with:
  - closed status strings: `PASS`, `PARTIAL`, `FAIL`, `STALE`, `N/A`, `UNREVIEWED`,
    `NOT_AVAILABLE`;
  - a current profile id: `i2i_edit_5x4_2026_06_05`;
  - `ValidationRecord`, `ValidationProfile`, and `ModelValidation`;
  - public lookup functions: `list_validation_profiles(...)`, `get_validation_profile(...)`, and
    `get_model_validation(...)`.
- Exported the validation API from `mflux` / `mlxgen`.
- Added `mlxgen validation`:
  - `mlxgen validation --list` lists available profiles;
  - `mlxgen validation` returns the current full validation profile;
  - `mlxgen validation --model <model>` accepts supported aliases and returns the matching exact
    model/package status rows.
- Kept `mlxgen capabilities` route-only and schema-compatible.
- Changed route-error wording from “validated capabilities” to “unified generation capabilities.”
- Regenerated the legacy `edit_capability_summary_contact_sheet.jpg` as the clear source-handle
  5x4 matrix instead of the old unreadable representative grid.
- Extended `edit_capability_manifest.md` generation so the exact-command manifest includes package
  variants displayed in the matrices, including FLUX.2 q4 rows and Qwen Image Edit 2511 q4 rows.
- Added tracked multi-reference input assets under `docs/assets/validation/.../reference-inputs/`
  so composition rows record the actual generated B/D reference images used by the validation prompt.

## Validation

- `uv run mlxgen validation --model briaai/Fibo-Edit` reports route-separated release status
  `FAIL`.
- `uv run mlxgen validation --model fibo-edit` resolves the alias and reports the same `FAIL`
  source rows.
- `uv run mlxgen validation --model AbstractFramework/qwen-image-edit-2509-4bit` reports
  `PARTIAL` with the multi-reference composition row marked `PARTIAL`.
- `uv run mlxgen capabilities --model briaai/Fibo-Edit` reports no unified generation
  capabilities.
- Focused tests:
  `uv run pytest tests/test_task_inference.py tests/cli/test_mlx_gen_router.py::test_capabilities_command_reports_model_modes tests/cli/test_mlx_gen_router.py::test_validation_command_reports_model_specific_status tests/cli/test_mlx_gen_router.py::test_validation_command_lists_profiles tests/cli/test_mlx_gen_router.py::test_routes_fibo_edit_plain_image_to_dedicated_edit_generation tests/cli/test_mlx_gen_router.py::test_fibo_edit_mask_path_is_not_advertised_by_unified_router tests/cli/test_mlx_gen_router.py::test_fibo_edit_masked_image_path_alias_is_not_advertised_by_unified_router -q`
  passed.
- `uv run ruff check` passed for the changed routing, validation, test, and report files.

## Residual risks

- The registry is static for the current profile. Future release gates should either update this
  registry directly or move the validation data to a tracked generated data file consumed by the
  registry.
- Raw `validation_outputs/` remains ignored. Curated proof contact sheets should be copied under
  `docs/assets/validation/` before a release if they need to ship with the repo or package docs.
- FIBO Edit still needs item [0027](../planned/0027_fibo_edit_diffusers_parity_release_quality.md)
  before it can be called release-quality for edit-reference I2I.

## Status Update - 2026-06-06

Item [0029](0029_qwen_image_edit_2511_base_parity.md) updated the registry with newer Qwen Image
Edit 2511 source/q8/q4 proof rows after the FlowMatch scheduler parity fix. FIBO Edit remains
unavailable through unified public capability discovery.
