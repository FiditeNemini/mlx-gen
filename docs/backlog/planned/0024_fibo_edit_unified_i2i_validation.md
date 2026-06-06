# Planned: FIBO Edit unified I2I validation

## Metadata
- Created: 2026-06-05
- Status: Planned
- Completed: N/A

## ADR status
- Governing ADRs: [ADR 0001](../../adr/0001_runtime_smoke_validation_for_model_routes.md), [ADR 0002](../../adr/0002_no_silent_automatic_fallbacks.md)
- ADR impact: None

## Context

Item [0023](../completed/0023_i2i_capability_validation_matrix.md) downloaded and tested
`briaai/Fibo-Edit` during the I2I validation matrix. The route executed, but the generated images
failed visual QA across masked minimal JSON, unmasked minimal JSON, full-schema JSON at 432x240,
and full-schema JSON at 640x352. The natural-language VLM prompt-conversion path also spent
minutes before denoising, which is not acceptable without explicit progress reporting.

This item was originally opened while unified `mlxgen capabilities` failed closed for FIBO Edit.
That is again the current release state: unified `mlxgen generate` exposes no public FIBO Edit
capability. The dedicated FIBO Edit command remains available only for parity testing.

The upstream `briaai/Fibo-Edit` model card identifies the package as `pipeline_tag: image-to-image`
with `image-editing` and `inpainting` tags, and describes a source-image plus optional-mask JSON
editing contract. So FIBO Edit is supposed to support editing in principle; MLX-Gen simply does not
yet have a passing local unified-route proof.

## Problem

FIBO Edit is an image-conditioned editor in the underlying model family, but current release
validation has not produced a passing source/BF16/q8 prepared-package proof for the standardized
spaceship edit sequence. Advertising it as a broad, release-quality generic I2I backend would let
applications auto-select a route that can produce unusable images.

## What we want to do

Repair or re-scope FIBO Edit support. Keep FIBO Edit unavailable through unified public capability
discovery until model-backed visual proof passes.

## Requirements

- Compare the MLX implementation against the upstream FIBO Edit inference contract.
- Confirm required prompt schema, VLM conversion behavior, scheduler, guidance, dimensions, VAE
  scaling, conditioning image packing, and mask handling.
- Add progress events around the VLM prompt-conversion phase if prose prompts remain supported.
- Validate with at least one plain edit, one masked edit, and one RMBG run if `Fibo-Edit-RMBG` is
  installed.
- Keep unified FIBO Edit capability unavailable until source-model visual proof passes.

## Non-goals

- Do not block the current installed-model I2I validation matrix.
- Do not claim generic inpaint/outpaint support; item 0019 tracks first-class fill/outpaint UX.

## Expected outcomes

- Either FIBO Edit has passing local proof for the relevant public surface, or docs clearly mark
  the unsupported/validation-limited surfaces.
- Failed prompt contracts are rejected early instead of producing garbage images.
- FIBO VLM preprocessing emits progress events or a clear pre-denoise status.

## Current progress - 2026-06-05

Six independent review charters were applied:

| Charter | Finding |
| --- | --- |
| Architect/platform | Keep FIBO Edit unavailable through unified capability discovery until model-backed proof passes. |
| Upstream contract | FIBO Edit expects JSON with `edit_instruction`; optional masks are folded into the conditioning image rather than used as a hard latent mask. |
| Text/transformer audit | Prompt padding masks were discarded before transformer attention, which could let padded prompt tokens influence generation. |
| CLI and routing audit | FIBO LoRA flags were accepted but ignored; base FIBO CLI could accept an edit config; edit JSON missing `edit_instruction` could fall back to VLM conversion. |
| Visual QA | Historical FIBO Edit rows from item 0023 were not acceptable visual proof. |
| Review/release gate | Do not claim release-quality FIBO Edit until true-handle source/BF16/q8 rows pass focused validation. |

Implemented fixes:

- `PromptEncoder.encode_prompt()` now returns a prompt attention mask, uses upstream-scale
  `max_sequence_length=3000`, and uses an empty negative prompt when none is supplied.
- FIBO transformer attention now consumes the prompt attention mask and rejects incompatible mask
  shapes instead of rebuilding an all-ones text mask.
- Base FIBO and FIBO Edit pass prompt attention masks into the denoiser.
- FIBO Edit default guidance is 5.0.
- JSON prompts that parse successfully but do not contain `edit_instruction` are rejected before
  generation instead of falling back to prose/VLM conversion.
- FIBO LoRA flags are rejected for generation and prepare workflows until a real FIBO LoRA mapping
  exists.
- FIBO generation rejects edit-only configs, and the FIBO Edit CLI reports missing/invalid prompt
  errors through the parser instead of a raw traceback.
- Unified route discovery exposes no public FIBO Edit route; current release validation still
  reports source/BF16/q8 FIBO Edit rows as failing.

Focused tests passed:

```sh
uv run pytest \
  tests/image_generation/test_fibo_edit_util.py \
  tests/image_generation/test_fibo_attention_mask.py \
  tests/arg_parser/test_cli_argparser.py::test_fibo_args \
  tests/arg_parser/test_cli_argparser.py::test_fibo_generate_rejects_edit_model_config \
  tests/arg_parser/test_cli_argparser.py::test_fibo_edit_args \
  tests/arg_parser/test_cli_argparser.py::test_fibo_edit_args_can_parse_without_prompt_for_runtime_validation \
  tests/arg_parser/test_cli_argparser.py::test_fibo_edit_rejects_non_edit_model_restored_from_metadata \
  tests/cli/test_prepare_save.py::test_prepare_rejects_fibo_lora \
  tests/cli/test_fibo_cli.py \
  tests/test_task_inference.py \
  tests/cli/test_mlx_gen_router.py::test_fibo_edit_mask_path_is_not_advertised_by_unified_router \
  tests/cli/test_mlx_gen_router.py::test_fibo_edit_masked_image_path_alias_is_not_advertised_by_unified_router \
  -q
```

Result: 35 passed.

Model-backed validation assets:

| Case | Source | Prompt | Output | Status |
| --- | --- | --- | --- | --- |
| Plain pencil edit | `validation_outputs/fibo_edit_repair_2026_06_05/source_spaceship_snow_768x432.png` | `validation_outputs/fibo_edit_repair_2026_06_05/plain_pencil_prompt.json` | `validation_outputs/fibo_edit_repair_2026_06_05/plain_pencil_672x384_seed8986.png` | Passed: coherent pencil sketch, source geometry preserved. |
| Masked engine edit | same source, `engine_mask_768x432.png` | `masked_engine_prompt.json` | `masked_engine_672x384_seed8987.png` | Partial: scene preserved, prompt color change weak. |
| Masked strict amber edit | same source, `engine_mask_768x432.png` | `masked_engine_prompt_strict_amber.json` | `masked_engine_strict_amber_672x384_seed8988.png` | Partial: amber improved, but engine area over-edited. |
| Tight masked strict amber edit | same source, `engine_mask_tight_768x432.png` | `masked_engine_prompt_tight_amber.json` | `masked_engine_tight_amber_672x384_seed8989.png` | Failed visual QA: scene over-edited and oversaturated. |

Contact sheet:

- `validation_outputs/fibo_edit_repair_2026_06_05/fibo_edit_repair_contact_sheet.jpg`

## Compatibility pass - 2026-06-05

Five additional review charters were applied after the first repair pass:

| Charter | Finding |
| --- | --- |
| Architect/platform | Re-enable only the narrow one-image unmasked `edit-reference` path if unified visual proof passes; keep masked FIBO and RMBG fail-closed. |
| Upstream parity | A later Diffusers parity audit found that empty CFG rows must encode to FIBO's begin-of-text token, not to a zero-length prompt. |
| Mask/conditioning math | FIBO mask handling is soft conditioning, not hard inpaint/outpaint preservation, so it must not be advertised through `supports_mask`. |
| CLI/API audit | `mlxgen prepare` must route FIBO Edit/RMBG through `FIBOEdit`; completion scripts must not advertise ignored LoRA flags. |
| Review/release gate | Require real unified `mlxgen generate` proof for default and q8 plain edit before exposing the capability. |

Additional fixes:

- `fibo-edit` now exposes no unified public generation capabilities after local Diffusers and MLX
  validation failed to produce acceptable edit images.
- `fibo-edit-rmbg` exposes no unified capabilities until local RMBG proof and a clean public
  contract exist.
- `--mask-path`, multiple `--image` inputs, and `--image-strength` are rejected for FIBO Edit
  before weight loading.
- FIBO Edit prompt parsing accepts Python dictionaries for direct API use.
- Masked natural-language prompt conversion is rejected early; masked FIBO tests require explicit
  JSON with `edit_instruction`.
- FIBO/FIBO Edit VLM prompt conversion prints a clear pre-denoise status in CLI flows.
- `mflux-generate-fibo-edit` exits non-zero on recoverable validation errors.
- FIBO Edit prepare uses the FIBO Edit model class instead of the base FIBO class.

Superseded validation note:

- Earlier alias/runtime-q8 proof accepted the pre-existing empty-negative behavior because one
  sketch-style row looked better locally. Item
  [0026](../completed/0026_edit_model_prepared_capability_contact_sheets.md) invalidated that as
  release evidence because it did not use the true source handle or prepared packages. A later
  Diffusers parity audit restored upstream-compatible empty CFG tokenization.

Historical unified proof assets:

| Case | Command shape | Output | Status |
| --- | --- | --- | --- |
| Plain FIBO Edit q8 | `uv run mlxgen generate --model fibo-edit --quantize 8 --image source_spaceship_snow_768x432.png --prompt-file plain_pencil_prompt.json --width 672 --height 384 --canvas-policy exact-resize --steps 50 --guidance 5 --seed 8986` | `validation_outputs/fibo_edit_repair_2026_06_05/unified_plain_pencil_q8_reverted_empty_672x384_seed8986.png` | Passed visual QA: coherent pencil sketch, source geometry preserved. |
| Plain FIBO Edit default weights | `uv run mlxgen generate --model fibo-edit --image source_spaceship_snow_768x432.png --prompt-file plain_pencil_prompt.json --width 672 --height 384 --canvas-policy exact-resize --steps 50 --guidance 5 --seed 8986` | `validation_outputs/fibo_edit_repair_2026_06_05/unified_plain_pencil_bf16_reverted_empty_672x384_seed8986.png` | Passed visual QA: coherent pencil sketch, source geometry preserved. |

Those rows prove the unified route can execute, but they are not release evidence for prepared
FIBO Edit packages because they used the short alias and runtime quantization. Current package
validation is tracked in
[0026](../completed/0026_edit_model_prepared_capability_contact_sheets.md) with the true source
handle `briaai/Fibo-Edit` and prepared local folders.

Current decision: keep this item planned for future masked FIBO Edit and RMBG work only. Plain
FIBO Edit remains unavailable through unified `mlxgen generate`; prepared-package visual status is
tracked separately in item 0026 and follow-up parity work is tracked in item 0027. Masked FIBO Edit
remains dedicated-command gated because visual QA still fails for localized mask edits, and
`briaai/Fibo-Edit-RMBG` still lacks a local release-quality proof.

## Porting correction - 2026-06-05

Diffusers parity review against
The local Diffusers Bria FIBO pipeline audit found two concrete MLX
porting issues:

- Empty negative prompt rows in FIBO CFG now encode as `<|begin_of_text|>` / token `128000`,
  matching Diffusers' FIBO prompt contract.
- FIBO transformer `norm_out.linear.bias` is now represented in the MLX module. Prepared FIBO
  folders created by older builds are rejected before generation if that weight is missing.

These fixes do not convert the historical failed images into passing evidence. Item
[0027](0027_fibo_edit_diffusers_parity_release_quality.md) tracks re-preparing FIBO Edit BF16/q8
folders with the corrected code and rerunning the standardized sequence.

## Correction - 2026-06-05

The compatibility pass above is not sufficient release evidence for q8 FIBO Edit packaging. It
used the `fibo-edit` alias and runtime `--quantize 8`; release validation must use the true source
handle `briaai/Fibo-Edit` and a prepared q8 folder or HF repo. Broader prepared-package edit
validation is now tracked in
[0026](../completed/0026_edit_model_prepared_capability_contact_sheets.md).

## Validation

- Focused unit tests for capability routing and prompt validation.
- Model-backed visual proofs for plain edit and masked edit.
- Failed-attempt assets preserved if a mode remains unsupported.
