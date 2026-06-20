# Planned: LoRA capability matrix and strict application

## Metadata

- Created: 2026-05-28
- Status: Planned
- Completed: N/A

## ADR status

- Governing ADRs: [ADR 0002](../../adr/0002_no_silent_automatic_fallbacks.md), [ADR 0003](../../adr/0003_runtime_truth_vs_consumer_convenience.md)
- ADR impact: May revise the generation capability contract. No new ADR is needed if LoRA remains
  task-specific capability metadata plus per-family adapter mappings. Escalate to an ADR only if
  MLX-Gen changes LoRA into a plugin/provider interface, stores adapters as a separate package
  class, or permits automatic fallback/substitution behavior.

## Context

MLX-Gen already exposes LoRA arguments and metadata, and the current public docs advertise LoRA
support. That support is real for some image families but not universal. As MLX-Gen becomes the
Apple Silicon backend for AbstractVision, callers need a reliable capability answer before they
offer LoRA controls in a UI or workflow.

Qwen-Image-Edit-2511 also makes LoRA strategically important: its model card calls out integrated
LoRA capabilities and community LoRA effects as part of the 2511 upgrade.

## Current code reality

- `src/mflux/cli/parser/parsers.py` adds `--lora-style`, `--lora-paths`, and `--lora-scales`.
- `src/mflux/task_inference.py` now exposes route-level LoRA fields in schema version 2:
  `supports_lora`, `lora_status`, `lora_target_roles`, and `lora_validation_profile`.
- Unified `mlxgen generate` derives `has_lora` from CLI flags or metadata and rejects unsupported
  LoRA routes before model dispatch.
- `src/mflux/models/common/resolution/lora_resolution.py` fails unresolved LoRA paths before model
  load and resolves cached LoRA repositories from both the MLX-Gen LoRA cache and the default
  Hugging Face cache.
- `LoraResolution.resolve_scales(...)` is strict: the number of scales must match the number of
  adapter paths exactly.
- `src/mflux/models/common/lora/mapping/lora_loader.py` raises `LoRAApplicationError` for missing
  files, unreadable files, corrupt files, zero matched keys, zero applied layers, missing A/B
  matrices, missing target paths, non-linear targets, and matrix shape mismatches.
- `src/mflux/models/common/lora/lora_compatibility.py` adds a cached model-card preflight for known
  adapter base-model mismatches. This currently rejects
  `lovis93/Flux-2-Multi-Angles-LoRA-v2` for FLUX.2 Klein because the adapter model card declares
  `black-forest-labs/FLUX.2-dev`.
- FLUX.2 and Qwen initializers call the compatibility preflight before loading model weights, and
  the unified router calls it before dispatch when a model config is resolved.
- LoRA mappings exist for FLUX.1, FLUX.2, Qwen, and Z-Image:
  - `src/mflux/models/flux/weights/flux_lora_mapping.py`
  - `src/mflux/models/flux2/weights/flux2_lora_mapping.py`
  - `src/mflux/models/qwen/weights/qwen_lora_mapping.py`
  - `src/mflux/models/z_image/weights/z_image_lora_mapping.py`
- FLUX.2 and Z-Image also have training adapters.
- ERNIE now has a real MLX-Gen LoRA path. `src/mflux/models/ernie_image/ernie_image_initializer.py`
  applies transformer LoRAs through `src/mflux/models/ernie_image/weights/ernie_image_lora_mapping.py`,
  and `mlxgen capabilities` surfaces ERNIE text/latent routes as LoRA-capable.
- Bonsai still accepts `lora_paths` in constructor signatures for prepare compatibility, but
  `src/mflux/models/bonsai_image/bonsai_image_initializer.py` deletes those arguments and sets
  `model.lora_paths = None`.
- SeedVR2 and FIBO do not have proven LoRA mappings in the current MLX-Gen tree. Wan has moved
  into its own completed rollout item: the runtime accepts Wan LoRAs and all current Wan q8 public
  rows now have exact validated route proofs. FIBO is already rejected when LoRA is requested, but
  it remains a useful negative test because FIBO Edit itself is currently deprioritized and
  unavailable through unified generation.
- `mlxgen prepare` parses LoRA flags for every model and currently rejects only FIBO explicitly.
  Signature-based forwarding prevented some constructor crashes, but it was not a capability
  contract and could still leave users with unsupported or ignored adapter requests on families that
  accept then discard LoRA kwargs. `mlxgen prepare --lora-paths/--lora-scales` is now rejected
  until save/reload LoRA bake behavior is explicitly proven for the selected family and
  quantization mode.
- `src/mflux/models/common/weights/saving/model_saver.py` bakes and strips LoRA wrappers before
  save. `src/mflux/models/common/lora/mapping/lora_saver.py` now fails on LoRA bake shape mismatch
  instead of silently skipping the delta. That keeps direct save/export calls aligned with ADR 0002
  while prepared-package LoRA baking remains gated.
- `docs/lora.md`, `docs/api.md`, and `docs/troubleshooting.md` document strict runtime LoRA
  behavior and the source/no-LoRA/with-LoRA validation method.
- `fal/Qwen-Image-Edit-2511-Multiple-Angles-LoRA` now applies to
  `AbstractFramework/qwen-image-edit-2511-8bit` through the public `mlxgen generate` route. The
  Qwen LoRA mapping accepts the Diffusers `transformer.transformer_blocks.*` key family and Qwen
  modulation-layer adapter keys. The first contact sheet is
  `docs/assets/validation/lora-2026-06-08/qwen2511-q8-multi-angle-lora-ab-contact-sheet.png`.
- `mlxgen capabilities` already surfaces `supports_lora`, `lora_status`, `lora_target_roles`, and
  `lora_validation_profile` on Qwen and FLUX.2 edit routes, including single-image edit,
  multi-reference, reframe, and outpaint-capable single-image edit rows where applicable.
- The current surfacing still stops at `mapped-unvalidated` plus `lora_validation_profile=null`,
  even for the first checked Qwen Image Edit 2511 q8 single-image edit proof. That gap is now
  closed for the first exact rows: `AbstractFramework/qwen-image-edit-2511-8bit` on `qwen.edit`,
  `AbstractFramework/qwen-image-edit-2509-8bit` on `qwen.edit`,
  `AbstractFramework/qwen-image-2512-8bit` on `qwen.text`,
  `AbstractFramework/z-image-turbo-8bit` on `z-image.text`,
  `AbstractFramework/ernie-image-turbo-8bit` on `ernie-image.text`, and
  `AbstractFramework/flux.2-klein-9b-8bit` on `flux2.edit`.
- The exact-row validation ids are now auditable end to end. `mlxgen capabilities` surfaces
  `lora_validation_profile`, and `src/mflux/release/validation_registry.py` now resolves the
  current exact LoRA profile ids through `mlxgen validation --profile <lora_validation_profile>`.
- `src/mflux/models/common/lora/mapping/lora_loader.py` now returns a structured
  `LoRAApplicationResult`, model initializers retain it, and generated output metadata records
  `lora_application_reports`, `lora_applied_file_count`, and `lora_applied_target_count`.
- A concrete Qwen Image Edit 2509 parity bug was fixed in
  `src/mflux/models/qwen/weights/qwen_lora_mapping.py`: the 2509 adapter family uses local
  modulation keys such as `transformer_blocks.{block}.img_mod.1.lora_A.default.weight` and
  `transformer_blocks.{block}.txt_mod.1.lora_B.default.weight`, while MLX-Gen previously only
  matched the 2511-style `transformer.transformer_blocks...` family. The corrected mapping now
  matches the full `1440/1440` tensors of `dx8152/Qwen-Edit-2509-Multiple-angles`.
- The first 2509 stacked validation only became credible after matching the public Lightning
  workflow profile as well: `8` steps and `guidance 1`. Earlier `guidance 4` runs looked broken
  even after the mapping fix and should not be treated as valid evidence for the 2509 route.
- The current 2509 q8 single-image edit proof is
  `docs/assets/validation/lora-2026-06-11/qwen2509_q8_multi_angle_ab_contact_sheet.png`. The
  corresponding metadata confirms `2` LoRA files applied, two `720`-target adapter applications
  (`1440` summed applied-target count), and `0` unmatched keys on the validated runs.
- Original `AbstractFramework/qwen-image-edit-8bit` now has an accepted exact q8 single-image edit
  proof. The current route-level validation uses the local `ghibli_style_qwen_v3.safetensors`
  adapter on same-seed edit trials, matches all `1680/1680` tensors, applies `840` targets, and
  is surfaced as validated on `qwen.edit` for the current release cycle.
- A second concrete Qwen mapping gap was exposed by the public
  `prithivMLmods/Qwen-Image-2512-Pixel-Art-LoRA` adapter. MLX-Gen already matched the attention and
  MLP tensors, but it initially missed the `diffusion_model.transformer_blocks.{block}.img_mod.1`
  and `txt_mod.1` modulation-key family. The corrected Qwen mapping now matches the full
  `1680/1680` tensors on `AbstractFramework/qwen-image-2512-8bit`, and that exact `qwen.text` row
  is now validated.
- ERNIE now has an exact public-route q8 proof with
  `reverentelusarca/ernie-image-elusarca-anime-style-lora`. The adapter matches all `504/504`
  tensors, applies `252` transformer targets, and produces the contact sheet
  `docs/assets/validation/lora-2026-06-11/ernie_turbo_q8_anime_style_ab_contact_sheet.png`.
- Remaining gaps are base Qwen Image validation, route-specific validation for Qwen
  multi-reference and reframe/outpaint rows, Z-Image latent img2img proof, ERNIE latent img2img
  proof, additional FLUX.2 package proofs beyond the 9B q8 single-image edit row, and first-class
  FLUX.2-dev support if the lovis multi-angle adapter is selected as the first FLUX.2-dev proof.
- The original `AbstractFramework/qwen-image-8bit` q8 package is now fully present locally and the
  exact-base `flymy-ai/qwen-image-realism-lora` adapter loads cleanly on that route
  (`480/480` matched keys, `240` targets applied). The remaining gap is validation quality rather
  than adapter compatibility: the current realism A/B has a clearly visible effect, but it also
  pulls framing and composition enough that the row should stay `mapped-unvalidated` until a better
  exact-base adapter or a stronger adherence-oriented profile is proven.

## Problem

LoRA should be treated as required user input, not best-effort decoration. If a user asks for a LoRA
and it is missing, corrupt, maps zero keys, or targets a family that does not support LoRA, MLX-Gen
should fail early with a clear message. Silent or warning-only behavior is dangerous because the
output image can look plausible while ignoring the requested adapter.

This is now planned because the current behavior is inconsistent with ADR 0002 and the user-facing
troubleshooting docs: resolution is strict, but application can still degrade silently later.

## What we want to do

Add a capability matrix and strict LoRA application policy in two phases:

Phase 1, strictness and introspection:

1. Add family-level capability metadata for LoRA inference and LoRA training.
2. Add task/mode-specific LoRA metadata to each `GenerationCapability` so callers can ask whether
   LoRA is supported for T2I, latent I2I, edit-reference I2I, multi-reference I2I, canvas-guided
   outpaint/reframe, T2V, or I2V.
3. Make the unified `mlxgen` router reject LoRA flags for unsupported families and unsupported
   task/mode combinations before model load.
4. Change LoRA loading so user-requested files must exist, load, and apply at least one mapped
   target.
5. Make `--lora-scales` strict: a scale list must match the adapter list exactly, and scales
   without adapters must fail before model load.
6. Keep partial-match warnings for valid adapters, but fail zero-match adapters by default.
7. Make `mlxgen prepare --lora-paths` fail closed unless the selected family and quantization mode
   has a tested, deterministic LoRA bake/export path.
8. Add docs and generated capability metadata so AbstractVision can decide whether to display LoRA
   controls.

Phase 2, visual support claims:

1. Select one known adapter per supported image family and task direction.
2. Produce model-backed A/B proofs with identical prompt, seed, dimensions, steps, guidance, and
   input image where relevant.
3. Promote capability status from "mapped but unvalidated" to "validated" only for the exact
   model family, mode, and package class that passed visual review.

Initial support matrix should be explicit and task-aware:

| Family | Current MLX-Gen LoRA status | Task directions | Difficulty | Path |
| --- | --- | --- | --- | --- |
| FLUX.1 | Inference mapping exists in the mflux-derived code, but unified `mlxgen capabilities` is not currently centered on FLUX.1. Fill, depth, control, redux, kontext, and in-context variants share related transformer concepts but need route-specific proof. | Dedicated CLI compatibility first; unified T2I/fill only after explicit capability work | Low for loader strictness; medium for unified proof | Do not advertise FLUX.1 LoRA through unified capabilities until FLUX.1 itself is deliberately revalidated. Dedicated CLI LoRA can remain compatibility behavior if it becomes strict and documented separately. |
| FLUX.2 Klein | Inference mapping exists; training adapter exists. | T2I, latent I2I, edit-reference, multi-reference | Low to medium | Keep supported, add strict loader tests and one visible T2I plus one I2I/edit validation row per representative package. |
| Qwen Image / Qwen Image Edit / Qwen Image Edit 2509 / 2511 | Inference mapping exists. Official Qwen Image Edit 2511 advertises integrated LoRA capability and many adapters exist in the HF model tree. Exact q8 validated rows now exist for original Qwen Image Edit, Qwen Image Edit 2509, and Qwen Image 2511 single-image edit, plus Qwen Image 2512 text-to-image. | T2I where the model supports it; I2I edit-reference and multi-reference for edit models | Medium | Keep supported only after visible validation with real Qwen LoRAs. Base Qwen Image still needs its own exact proof, and Qwen multi-reference plus reframe/outpaint rows remain separate validations because prompt/image contracts differ. |
| Z-Image / Z-Image Turbo | Inference mapping and training adapter exist; one public LoRA slow test exists. | T2I, latent I2I where routed | Low to medium | Keep supported, make strict failure behavior common with FLUX/Qwen, and preserve visible LoRA regression output. |
| ERNIE Image / Turbo | Runtime mapping exists and exact q8 text-to-image proof exists for `AbstractFramework/ernie-image-turbo-8bit`. | T2I, latent I2I | Medium | Keep text-to-image supported with strict loader behavior. Treat latent img2img as `mapped-unvalidated` until a source-preserving proof lands. |
| Bonsai | Initializer still ignores LoRA; packed ternary/low-bit layout is not a normal adapter target and the current public “Bonsai LoRA” candidate uses SDXL UNet keys. | T2I | High / architectural | Reject LoRA flags. Revisit only through separate proposed item 0038. |
| FIBO / FIBO Edit | LoRA is rejected today; no proven mapping. FIBO Edit is deprioritized and not a release-quality unified edit route. | T2I only for base FIBO; FIBO Edit disabled in unified generation | High / deferred | Keep rejected. Do not spend LoRA work here until base FIBO/FIBO Edit priority changes. |
| Wan2.2 TI2V/T2V/I2V | Runtime mapping, CLI support, target-role metadata, and generated-video metadata exist. Current q8 Wan rows are now validated route-by-route for TI2V-5B text-to-video, TI2V-5B first-frame image-to-video, T2V-A14B text-to-video, and I2V-A14B first-frame image-to-video. | T2V, I2V | Medium to high | Runtime/proof work is recorded in [completed item 0033](../completed/0033_video_lora_for_t2v_i2v.md). Keep future Wan follow-ups scoped to new package variants, new adapters, or a second video family rather than reopening the base route. |
| SeedVR2 | No LoRA mapping; current route is restoration/upscale rather than generation. | Image restoration/upscale today; video restoration proposed in item 0032 | Low value / not priority | Reject LoRA flags. Treat model-specific restoration controls such as resolution and softness separately from LoRA. |

Task-direction roadmap:

| Direction | Near-term stance | Notes |
| --- | --- | --- |
| T2I | Support for FLUX.2, Qwen Image, and Z-Image after strict application tests. FLUX.1 stays dedicated-CLI or revalidation-gated unless unified FLUX.1 support is deliberately restored. | This is the easiest surface because one prompt produces one image and visible adapter effects are easy to compare. |
| Latent I2I | Support only for families whose latent route uses the same mapped transformer and has source-preserving visual proof. | A LoRA that works for T2I can still overpower the encoded source when `--image-strength` is high. |
| Edit-reference I2I | Support only after a single-image edit proof shows both adapter effect and prompt adherence. | Qwen and FLUX.2 edit paths need separate validation from latent I2I. |
| Multi-reference I2I | Support only after a two-reference proof shows adapter effect without losing the intended reference composition. | Validate separately for Qwen 2509/2511 and FLUX.2 because their multi-reference contracts differ. |
| Canvas reframe/outpaint | Support only after item 0019's canvas route and a LoRA A/B proof pass for the specific family. | LoRA may amplify source-window drift; treat it as a separate validation row from normal edit-reference I2I. |
| T2V | Wan q8 route proof exists for TI2V-5B and T2V-A14B. | See [completed item 0033](../completed/0033_video_lora_for_t2v_i2v.md) for the exact validated Wan video rows. |
| I2V | Wan q8 route proof exists for TI2V-5B first-frame I2V and I2V-A14B. | Same as T2V; future work is about extra package variants or future video families, not the current Wan route. |

## Suggested implementation

Make the capability contract the source of truth:

- Extend `GenerationCapability` with LoRA fields such as:
  - `supports_lora: bool`;
  - `lora_status: "unsupported" | "mapped-unvalidated" | "validated"`;
  - `lora_validation_profile: str | None`;
  - `lora_target_roles: tuple[str, ...]` for future multi-transformer/video models.
- Keep capability metadata task/mode specific. For example, `qwen.edit` and
  `qwen.multi-reference` can have different LoRA validation status even though they share the
  same handler.
- Add `has_lora` to `resolve_generation_plan(...)` and `resolve_task(...)`, derived by the CLI
  and Python callers from `lora_paths`, `lora_scales`, or `lora_style`.
- Reject unsupported LoRA requests before loading weights. Error text should name the selected
  model, resolved public task/mode, and the closest supported alternatives when any exist.
- Add a structured loader result, for example `LoRAApplicationReport`, with adapter path, scale,
  matched key count, applied target count, unmatched key count, and target roles. Save it in
  generated metadata.
- Add a dedicated `LoRAApplicationError` or equivalent `ValueError` for unreadable files, corrupt
  files, zero matched keys, zero applied targets, missing A/B matrices, target-path misses, and
  matrix shape mismatches.
- Treat `mlxgen prepare` as a separate contract from runtime LoRA loading. Runtime LoRA can wrap
  linear layers at generation time; prepared-package LoRA baking must either prove output
  equivalence after save/reload or fail with a clear message. In particular, q4/q8 packed
  linears must not skip LoRA deltas silently.
- If a LoRA is baked into a saved package, record original adapter paths, resolved files, scales,
  bake status, target counts, and any quantization constraints in package metadata and the model
  card.
- Keep adapter downloads explicit. `mlxgen download --model <lora-repo> --all-files` can prepare
  cache state; generation must not auto-download LoRA files.
- Do not make `--lora-scales` a standalone behavior. If scales are present and paths are absent,
  fail with a parser error.

## Why it might matter

LoRA is one of the fastest ways to make MLX-Gen useful for personal styles, product references,
characters, and AbstractVision workflows. It is also one of the easiest places to lie accidentally:
loading a model and ignoring the adapter creates outputs that are technically valid but
semantically wrong.

## Scope

- Family-level LoRA capability metadata for generation and training surfaces.
- Task-direction metadata for whether a family supports LoRA in T2I, latent I2I, edit-reference
  I2I, multi-reference I2I, fill/inpaint/outpaint, T2V, or I2V.
- Strict router or model-load rejection for families without proven LoRA inference support.
- Strict loader behavior for unreadable, corrupt, zero-match, or shape-invalid user-requested
  adapters.
- Strict scale-count validation and a clear error for scales without adapter paths.
- Generated metadata that records which adapters were actually applied, their scales, and how many
  targets matched.
- Promotion policy follow-up for validated rows: exact public validation profiles should only be
  promoted when the proving adapter set reports `unmatched_key_count == 0` on the validated run.
- Prepared-package policy for LoRA: either tested bake/export for the selected family and
  quantization mode, or early rejection before loading/saving.
- Documentation cleanup for inherited/model-local READMEs that currently make broader LoRA claims
  than the unified router can prove.
- Docs and tests that align the public troubleshooting claim with runtime behavior.

## Non-goals

- Do not implement LoRA for every family.
- Do not extend Wan/video LoRA in this item; preserve that work in completed item 0033 and any future follow-up item that targets a new video family or package class.
- Do not add automatic LoRA downloads during generation.
- Do not change existing quantization policies.
- Do not bake LoRAs into prepared packages unless that behavior is explicitly documented, tested,
  and output-equivalent after save/reload for the relevant quantization mode.

## Validation

- Unit test missing file, corrupt file, zero matched keys, partial matched keys, and successful
  application.
- Unit test strict scale behavior: too few scales, too many scales, and scales without adapter
  paths fail instead of padding or truncating.
- Router tests proving LoRA is accepted only for capabilities marked as LoRA-supported and rejected
  for Bonsai, FIBO, SeedVR2, and unsupported modes. Wan now needs positive route tests plus exact
  validation-status tests rather than blanket unsupported-family assertions.
- `mlxgen prepare` tests proving unsupported LoRA is rejected for Bonsai, SeedVR2, and FIBO,
  instead of relying on constructor signatures or silently ignored kwargs.
- Prepared-package tests proving any claimed LoRA bake works after save/reload. For q4/q8, this
  must include an output-equivalence or strong latent/weight-effect check, not only shard count or
  file-size checks.
- `mlxgen capabilities --model ...` tests proving LoRA fields are visible per mode.
- Python API tests proving `resolve_generation_plan(... has_lora=True)` rejects unsupported modes
  before model instantiation.
- Real-image contact sheet with Qwen-Image-Edit-2511 and at least one public 2511 LoRA. The first
  single-image edit proof uses `fal/Qwen-Image-Edit-2511-Multiple-Angles-LoRA` on
  `AbstractFramework/qwen-image-edit-2511-8bit`. Multi-reference LoRA is not claimed yet.
- Real-image proof for each task direction marked supported:
  - T2I: one model-backed row per supported image family;
  - latent I2I: one source-preserving variation row where claimed;
  - edit-reference/multi-reference: one focused Qwen or FLUX.2 edit row where claimed;
  - canvas outpaint: after item 0019 route ownership is stable for the target model family;
  - native fill/inpaint outpaint: only after item 0019 or a follow-up item provides a validated
    fill/mask backend.
- Metadata test proving generated images preserve original LoRA paths and scales.
- Metadata test proving generated images record applied target counts, not only requested paths.
- Metadata tests for FLUX.2 edit-reference and multi-reference outputs, because those routes
  currently apply transformer LoRA through initialization but must still preserve LoRA provenance
  in generated output metadata.
- Save/prepare test proving baked or stripped LoRA behavior is documented, deterministic, and not
  skipped on packed q4/q8 weights.

Recommended first visual proof set:

| Family | Minimal proof |
| --- | --- |
| Z-Image Turbo | Existing public Z-Image Turbo LoRA A/B row, because public LoRA examples already exist and the model path is lightweight compared with Qwen. |
| FLUX.2 Klein | One public FLUX.2 Klein LoRA A/B row for T2I and one edit-reference row, preferably q8 package first. |
| Qwen Image Edit 2511 | One public 2511 adapter row for single-image edit and one multi-reference row. If the adapter is an acceleration LoRA rather than style/content LoRA, validate both speed/step contract and visual quality. |
| Qwen Image / Qwen Image Edit / 2509 | Validate separately before marking supported because base generation, 2509, and 2511 use different prompt/image contracts. |

## Progress checklist

- [x] Confirm unresolved LoRA paths now fail in `LoraResolution`.
- [x] Confirm loader-level missing/unreadable and zero-match cases previously avoided hard failure.
- [x] Add explicit family-level LoRA capability metadata.
- [x] Add task-direction LoRA metadata so UI/API callers know which modes can accept adapters.
- [x] Add `has_lora` planning input and route-level rejection before model load.
- [x] Make `--lora-scales` fail when the count differs from `--lora-paths`, and fail when scales
      are provided without paths.
- [x] Reject LoRA flags for unsupported families before generation starts.
- [x] Add cached model-card base-model preflight for known incompatible adapters.
- [x] Reject or prove `mlxgen prepare --lora-paths` for unsupported families and q4/q8 packed
      packages; no skipped-bake prepared package may be saved as if LoRA was applied.
- [x] Make loader-level missing, unreadable, corrupt, zero-match, and shape-invalid cases fail
      closed.
- [x] Return and persist a structured LoRA application report.
- [x] Promote exact proven route/package rows from `mapped-unvalidated` to `validated`, starting
      with the Qwen Image Edit 2511 q8 single-image edit row only if its capabilities metadata can
      name that proof without accidentally promoting multi-reference or other package classes.
- [x] Add focused tests for strict LoRA application.
- [x] Add first model-backed A/B contact sheet for Qwen Image Edit 2511 q8.
- [x] Add first model-backed A/B contact sheets for Z-Image Turbo and FLUX.2 Klein before marking
      those exact modes validated. The downloaded lovis93 multi-angle adapter targets FLUX.2-dev,
      so it remains tracked separately from FLUX.2 Klein.
- [x] Update docs and capability metadata for the current mapped-unvalidated contract.

## Guidance for future agents

Start with strict diagnostics and the capability matrix before adding new family mappings. If a
family cannot prove LoRA application with a known adapter and visible output difference, mark it as
unsupported rather than "probably works".

## Sources checked

- `src/mflux/models/common/lora/mapping/lora_loader.py`
- `src/mflux/models/qwen/weights/qwen_lora_mapping.py`
- `src/mflux/models/flux2/weights/flux2_lora_mapping.py`
- `src/mflux/models/z_image/weights/z_image_lora_mapping.py`
- Qwen-Image-Edit-2511 model card: https://huggingface.co/Qwen/Qwen-Image-Edit-2511
- Qwen-Image-Edit-2509 model card: https://huggingface.co/Qwen/Qwen-Image-Edit-2509
- FLUX.2 Klein 4B model card: https://huggingface.co/black-forest-labs/FLUX.2-klein-4B
- Wan2.2 TI2V-5B Diffusers model card: https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B-Diffusers
- Public Z-Image Turbo LoRA example: https://huggingface.co/renderartist/Classic-Painting-Z-Image-Turbo-LoRA
