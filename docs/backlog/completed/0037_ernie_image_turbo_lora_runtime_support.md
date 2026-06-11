# Completed: ERNIE Image Turbo LoRA runtime support

## Metadata

- Created: 2026-06-11
- Status: Completed
- Completed: 2026-06-11

## ADR status

- Governing ADRs: [ADR 0002](../../adr/0002_no_silent_automatic_fallbacks.md)
- ADR impact: None

## Context

`0007` established the strict LoRA contract, but ERNIE Image Turbo was still fail-closed in
MLX-Gen even though upstream Diffusers already exposes `ErnieImageLoraLoaderMixin`. The user wanted
an explicit feasibility answer for ERNIE and Bonsai rather than another generic “unsupported”
statement.

## Current code reality

- `src/mflux/models/ernie_image/ernie_image_initializer.py` used to accept `lora_paths` and
  `lora_scales` but immediately discard them.
- `src/mflux/task_inference.py` exposed ERNIE text and latent routes with `supports_lora=false`.
- Upstream Diffusers already normalizes ERNIE LoRA keys under `diffusion_model.*` and
  `transformer.*`.
- Public adapter `reverentelusarca/ernie-image-elusarca-anime-style-lora` uses
  `diffusion_model.layers.{layer}.self_attention.*` and `mlp.*` keys that line up with MLX-Gen’s
  ERNIE transformer structure.

## Problem

ERNIE LoRA support was blocked by missing MLX wiring, not by a missing adapter ecosystem or an
unresolved architecture question. Keeping the route fail-closed would leave a proven upstream
capability unavailable in MLX-Gen.

## What we changed

- Added an ERNIE-specific LoRA mapping in
  `src/mflux/models/ernie_image/weights/ernie_image_lora_mapping.py`.
- Updated `src/mflux/models/ernie_image/ernie_image_initializer.py` to validate adapter
  compatibility, apply LoRA to the ERNIE transformer, and retain structured loader reports in model
  state and metadata.
- Updated `src/mflux/models/ernie_image/cli/ernie_image_generate.py` and
  `src/mflux/models/ernie_image/variants/ernie_image_turbo.py` so the public ERNIE CLI and
  `mlxgen generate` route can pass LoRA inputs and preserve LoRA metadata.
- Extended `src/mflux/models/common/lora/lora_compatibility.py` with ERNIE model-class handling.
- Promoted ERNIE text-to-image LoRA support in `src/mflux/task_inference.py` and
  `src/mflux/lora_validation_registry.py`.

## Expected outcomes

- `mlxgen generate --model AbstractFramework/ernie-image-turbo-8bit --lora-paths ...` works through
  the public route.
- ERNIE text-to-image capability rows surface `supports_lora=true`.
- Exact validated ERNIE rows report `lora_status="validated"` and an exact
  `lora_validation_profile`.
- Latent ERNIE img2img remains `mapped-unvalidated` until a separate source-preserving proof lands.

## Validation

- Focused tests:

  ```sh
  uv run pytest tests/test_ernie_lora_mapping.py tests/test_lora_compatibility.py tests/test_task_inference.py -q
  ```

  Result: `40 passed`

- Public-route proof:

  ```sh
  uv run mlxgen generate \
    --model AbstractFramework/ernie-image-turbo-8bit \
    --prompt "elusarca anime style, a young woman with silver hair and a red trench coat standing beneath glowing lanterns in a rain-soaked alley at night, confident pose, detailed face, dramatic lighting" \
    --negative "blurry, deformed face, extra limbs, text, watermark" \
    --width 512 \
    --height 512 \
    --steps 8 \
    --guidance 1 \
    --seed 9961 \
    --metadata \
    --replace \
    --output validation_outputs/lora_strict_2026_06_11/ernie_turbo_q8_no_lora_anime.png

  uv run mlxgen generate \
    --model AbstractFramework/ernie-image-turbo-8bit \
    --prompt "elusarca anime style, a young woman with silver hair and a red trench coat standing beneath glowing lanterns in a rain-soaked alley at night, confident pose, detailed face, dramatic lighting" \
    --negative "blurry, deformed face, extra limbs, text, watermark" \
    --width 512 \
    --height 512 \
    --steps 8 \
    --guidance 1 \
    --seed 9961 \
    --metadata \
    --replace \
    --output validation_outputs/lora_strict_2026_06_11/ernie_turbo_q8_with_lora_anime.png \
    --lora-paths reverentelusarca/ernie-image-elusarca-anime-style-lora:ernie-anime-v1.safetensors \
    --lora-scales 0.9
  ```

- Visual proof asset:
  `docs/assets/validation/lora-2026-06-11/ernie_turbo_q8_anime_style_ab_contact_sheet.png`
- Loader metadata proof:
  `validation_outputs/lora_strict_2026_06_11/ernie_turbo_q8_with_lora_anime.metadata.json`
  reports `504/504` matched keys, `0` unmatched keys, and `252` applied targets.

## Completion report

### Date

2026-06-11

### Summary

ERNIE Image Turbo now has real MLX-Gen runtime LoRA support on the public text-to-image route, with
strict loader behavior, exact capability surfacing, and a model-backed q8 validation row.

### Files touched

- `src/mflux/models/ernie_image/weights/ernie_image_lora_mapping.py`
- `src/mflux/models/ernie_image/ernie_image_initializer.py`
- `src/mflux/models/ernie_image/cli/ernie_image_generate.py`
- `src/mflux/models/ernie_image/variants/ernie_image_turbo.py`
- `src/mflux/models/common/lora/lora_compatibility.py`
- `src/mflux/task_inference.py`
- `src/mflux/lora_validation_registry.py`
- `tests/test_ernie_lora_mapping.py`
- `tests/test_lora_compatibility.py`
- `tests/test_task_inference.py`
- `docs/lora.md`
- `docs/backlog/planned/0007_lora_capability_matrix_and_strict_application.md`
- `docs/backlog/overview.md`

### Residual risk

- Only `AbstractFramework/ernie-image-turbo-8bit` text-to-image is validated exactly.
- ERNIE latent img2img now has a mapping path but still needs a separate source-preserving proof.
- Non-turbo ERNIE remains outside the current validation set.

### Follow-up

- Keep ERNIE latent img2img under `0007` until it has its own proof row.
- Track Bonsai separately because its packed runtime is a different architectural problem.
