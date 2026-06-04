# Completed: taskless generation routing

## Metadata
- Created: 2026-06-04
- Status: Completed
- Completed: 2026-06-04

## ADR status
- Governing ADRs: ADR 0002
- ADR impact: None. This item implements the fail-closed task-resolution policy already required by ADR 0002.

## Context
`mlxgen generate` should choose the generation task from the selected model and input images when the
request is unambiguous. Users should not need `--task image-to-video` for an I2V-only Wan model with
one input image, or `--task text-to-image` for an image model without one.

The task flag remains useful as an explicit override for edit workflows or for compatibility, but it
should not be required for the common T2I, I2I, T2V, and I2V paths.

## Current code reality
- `src/mflux/cli/mlx_gen.py` already defaulted `--task` to `auto`, but route resolution mixed family
  detection, model overrides, image cardinality, and task validation in one function.
- FLUX.2 routed a single `--image` to the edit backend unless `--image-strength` or
  `--task image-to-image` was supplied.
- Qwen and FIBO accepted some contradictory task/image combinations, such as
  `--task image-to-image` without an image or `--task text-to-image` with an image.
- Wan configs already declared task facts in `ModelConfig.transformer_overrides`: TI2V supports both
  text-to-video and image-to-video, T2V-A14B is text-to-video only, and I2V-A14B is image-to-video
  only.
- Wan runtime validated unsupported task/image combinations after model construction, which could
  load very large weights before rejecting an impossible request.
- Python callers instantiate family classes directly today. There was no small public task resolver
  they could call to apply the same inference contract used by the CLI.

## Problem
The CLI and Python-facing behavior made `--task` feel necessary even when the model and image inputs
already identified the task. Some contradictory inputs were silently accepted, which risked expensive
wrong-backend runs and conflicted with ADR 0002.

## What we wanted to do
Make task inference explicit, public, and fail-closed:

- no image means T2I or T2V for models that support those tasks;
- one image means I2I or I2V for models that support those tasks;
- multiple images mean edit only for edit-capable image backends;
- model-fixed tasks, such as Wan A14B T2V/I2V, reject incompatible image inputs before model load;
- `--task` remains an explicit override but cannot contradict the supplied images.

## Outcome
- Added public task inference in `src/mflux/task_inference.py`, exported through `mlxgen`.
- Wired `src/mflux/cli/mlx_gen.py` to use the shared resolver before backend dispatch.
- Changed FLUX.2 single-image routing to image-to-image; explicit `--task edit` and multiple images
  still route to edit.
- Added early Wan A14B fixed-task validation before backend/model load.
- Made generic unresolved Wan-like local names fail closed in the public resolver.
- Kept image progress task labels aligned with the real positive-`image_strength` img2img path.
- Removed `--task text-to-video` and `--task image-to-video` from normal Wan examples and generated
  Wan model-card usage snippets.

## Non-goals preserved
- Did not build a full high-level `mlxgen.generate(...)` facade.
- Did not change model weight formats, quantization, prompt conditioning, or denoising logic.
- Did not remove `--task` entirely; it remains an explicit override.
- Did not add a separate model capability JSON registry.

## Validation
- `uv run pytest tests/cli/test_mlx_gen_router.py tests/test_task_inference.py tests/callbacks/test_progress_callbacks.py tests/model_saving/test_model_card_saver.py -q`
  passed with 86 tests.
- `uv run ruff check src/mflux/task_inference.py src/mflux/cli/mlx_gen.py src/mflux/callbacks/generation_context.py tests/cli/test_mlx_gen_router.py tests/test_task_inference.py tests/callbacks/test_progress_callbacks.py tests/model_saving/test_model_card_saver.py`
  passed.
- A broader `uv run pytest -q` completed with 701 passed, 7 skipped, and 37 failures in
  model/resource-heavy areas outside this task: cached Depth Pro weight loading, model-backed image
  generation, training preview generation, Wan initializer/generation mock paths, and download
  policy tests.

## Completion checklist
- [x] Add planned backlog item and overview entry.
- [x] Implement shared task resolver.
- [x] Wire router to resolver and early Wan validation.
- [x] Update tests and docs.
- [x] Run focused validation and close the item when complete.

## Residual risks
- Direct backend CLIs and direct Python model-class calls can still bypass the top-level router.
  That is acceptable for this item because the new public resolver gives applications a shared
  contract without introducing a full generation facade.
- `--task` remains as an override, so future docs should continue to teach it only for explicit edit
  or compatibility cases, not normal T2I/I2I/T2V/I2V routing.
