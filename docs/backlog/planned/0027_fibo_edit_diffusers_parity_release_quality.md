# Planned: FIBO Edit Diffusers parity and release-quality validation

## Metadata
- Created: 2026-06-05
- Status: Planned
- Completed: N/A

## ADR status
- Governing ADRs: [ADR 0001](../../adr/0001_runtime_smoke_validation_for_model_routes.md), [ADR 0002](../../adr/0002_no_silent_automatic_fallbacks.md)
- ADR impact: None

## Context

Completed item [0026](../completed/0026_edit_model_prepared_capability_contact_sheets.md)
validated true `briaai/Fibo-Edit` source weights plus local BF16 and q8 prepared folders against
the same edit sequence used for Qwen and FLUX.2. The route executed, but FIBO Edit did not pass
release-quality visual validation for the spaceship edit sequence.

The follow-up Diffusers parity audit found concrete MLX-Gen issues:

- Empty negative prompts in FIBO CFG must encode as the FIBO begin-of-text token, matching
  Diffusers' Bria FIBO pipeline.
- FIBO's final AdaLN projection includes `norm_out.linear.bias`; the old MLX module dropped that
  bias because it reused a Flux helper with `bias=False`.
- Single-attention masks must broadcast over all attention heads before MLX scaled dot-product
  attention.
- Diffusers clips FP16 FIBO block activations to the finite half-precision range; MLX now mirrors
  that path when tensors are `float16`.
- SmolLM3 text encoder layers follow the upstream `no_rope_layers` pattern; MLX previously applied
  RoPE in every layer.
- Old local prepared FIBO Edit folders were saved without `norm_out.linear.bias` and must be
  regenerated before they can be used as current release evidence.

## Current code reality

- Unified `mlxgen generate` routes `briaai/Fibo-Edit` and FIBO Edit prepared paths to one-image
  unmasked `image-to-image` / `edit-reference`.
- FIBO Edit does not expose multi-reference, `--image-strength`, outpaint, or unified mask support.
- FIBO prompt encoding now represents empty prompt rows with `<|begin_of_text|>` / token `128000`.
- FIBO transformer `norm_out` now has a bias parameter, and FIBO weight loading rejects transformer
  weights that do not include `norm_out.linear.bias`.
- FIBO single-attention masks now broadcast to the configured head count.
- FIBO joint and single transformer blocks now include Diffusers-style FP16 finite-range clipping.
- FIBO SmolLM3 attention now respects the upstream NoPE/RoPE layer pattern.
- Existing validation images in `validation_outputs/edit_prepared_capability_2026_06_05/` are
  useful as failure evidence, not as passing proof.
- A source-handle rerun after the first parity fixes still failed the standardized pencil/crash row.
  The latest saved source artifact preserves some spaceship structure but remains overexposed and
  does not satisfy the crash/sketch edit:
  `validation_outputs/edit_prepared_capability_2026_06_05/fibo_edit_source_d_pencil_crash_after_nope_fix_672x384_50s_seed9433.png`.
- Current local prepared folders `models/fibo-edit-bf16` and `models/fibo-edit-8bit` contain
  `norm_out.linear.bias`; the q8 folder also keeps the q8-sensitive transformer paths unquantized.
  They still do not provide passing release evidence.
- The latest current-folder failure is numerical: BF16 and q8 prepared runs hit non-finite decoded
  image tensors on the 50-step crash profile. Instrumentation localized the first non-finite latent
  stream to denoise step index 30, after FIBO joint transformer block 7. Treat this as unresolved
  transformer math parity, not a packaging-only problem.

## Problem

FIBO Edit is currently removed from unified public capabilities. Local Diffusers and MLX FIBO Edit
runs did not produce acceptable images in the current validation environment, so `mlxgen generate`
must fail closed for FIBO Edit until source-model parity and visual validation pass.

## What we want to do

Bring FIBO Edit to a defensible release-quality state or keep it unavailable through unified
capability discovery.

## Requirements

- First build tensor-level parity against Diffusers for the source route. Re-prepare FIBO Edit
  BF16 and q8 folders from `briaai/Fibo-Edit` after any transformer math fix, then rerun validation.
- Re-run the standardized source/BF16/q8 edit sequence with explicit JSON prompts, dimensions,
  steps, guidance, seed, source image, and output paths.
- Compare the current MLX route against Diffusers' `BriaFiboEditPipeline` for:
  - empty negative prompt tokenization;
  - final AdaLN bias loading;
  - scheduler shift constants and timestep grid;
  - BF16/FP16 handling and any required fp16 clipping;
  - VAE encode/decode scaling and temporal/cache behavior;
  - local prose-to-JSON model choice versus upstream FIBO edit prompt-to-JSON tooling.
- Keep old prepared folders from being used silently if required FIBO weights are missing.
- Produce a reviewer-readable matrix using the same source/pencil/crash/composition evidence style
  as item 0026.

## Non-goals

- Do not add multi-reference support to FIBO Edit.
- Do not claim masked edit, RMBG, inpaint, outpaint, or reframing support here.
- Do not publish FIBO-derived prepared folders unless license/gating policy is explicitly approved.

## Expected outcomes

- Either FIBO Edit source/BF16/q8 passes the standardized release-quality sequence, or public docs
  and capability discovery keep it unavailable.
- Prepared FIBO Edit folders generated with older MLX-Gen builds fail clearly instead of generating
  with missing final bias; current folders still fail until transformer parity is fixed.
- The final report lists exact commands, prompts, source image, package handles/paths, outputs, and
  pass/fail status for every validated row.

## Validation

- Focused unit tests:
  - FIBO empty prompt rows use BOT token `128000`;
  - FIBO transformer `norm_out` exposes a bias;
  - FIBO attention masks broadcast over heads;
  - FIBO FP16 activations clip to Diffusers' finite range;
  - FIBO SmolLM3 attention respects the upstream NoPE/RoPE layer pattern;
  - old prepared FIBO transformer weights missing `norm_out.linear.bias` are rejected.
- Tensor-level source-route parity against local Diffusers before any package validation:
  token IDs/masks, hidden layers, initial latents, conditioning latents/image IDs, scheduler
  sigmas/timesteps, first transformer output, first scheduler step, and decoded image range.
- After source-route parity passes:
  - `uv run mlxgen prepare --model briaai/Fibo-Edit --path models/fibo-edit-bf16`
  - `uv run mlxgen prepare --model briaai/Fibo-Edit --path models/fibo-edit-8bit -q 8`
- Serial `mlxgen generate` validation for source, BF16 prepared, and q8 prepared folders.
- Manual visual review contact sheet.
