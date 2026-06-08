# Planned: LoRA capability matrix and strict application

## Metadata

- Created: 2026-05-28
- Status: Planned
- Completed: N/A

## ADR status

- Governing ADRs: [ADR 0002](../../adr/0002_no_silent_automatic_fallbacks.md)
- ADR impact: None for the immediate strictness task. Escalate to a new ADR only if MLX-Gen
  changes LoRA into a plugin/provider interface rather than per-family mappings.

## Context

MLX-Gen already exposes LoRA arguments and metadata, and the current public docs advertise LoRA
support. That support is real for some image families but not universal. As MLX-Gen becomes the
Apple Silicon backend for AbstractVision, callers need a reliable capability answer before they
offer LoRA controls in a UI or workflow.

Qwen-Image-Edit-2511 also makes LoRA strategically important: its model card calls out integrated
LoRA capabilities and community LoRA effects as part of the 2511 upgrade.

## Current code reality

- `src/mflux/cli/parser/parsers.py` adds `--lora-style`, `--lora-paths`, and `--lora-scales`.
- `src/mflux/models/common/resolution/lora_resolution.py` now fails unresolved LoRA paths before
  model load, and `docs/troubleshooting.md` tells users that requested LoRAs are required.
- `src/mflux/models/common/lora/mapping/lora_loader.py` still has loader-level silent-degradation
  paths: missing or unreadable files print an error and return from `_apply_single_lora()`, and
  zero-match adapters can finish with warnings rather than a failed generation.
- `src/mflux/task_inference.py` exposes generation capabilities for tasks, modes, masks,
  outpaint, and image strength, but it does not yet expose task-specific LoRA support.
- LoRA mappings exist for FLUX.1, FLUX.2, Qwen, and Z-Image:
  - `src/mflux/models/flux/weights/flux_lora_mapping.py`
  - `src/mflux/models/flux2/weights/flux2_lora_mapping.py`
  - `src/mflux/models/qwen/weights/qwen_lora_mapping.py`
  - `src/mflux/models/z_image/weights/z_image_lora_mapping.py`
- FLUX.2 and Z-Image also have training adapters.
- ERNIE and Bonsai accept `lora_paths` in constructor signatures for prepare compatibility, but
  their initializers delete those arguments and set `model.lora_paths = None`.
- Wan, SeedVR2, and FIBO do not have proven LoRA mappings in the current MLX-Gen tree. FIBO is
  already rejected when LoRA is requested, but it remains a useful negative test because FIBO Edit
  itself is currently deprioritized and unavailable through unified generation.

## Problem

LoRA should be treated as required user input, not best-effort decoration. If a user asks for a LoRA
and it is missing, corrupt, maps zero keys, or targets a family that does not support LoRA, MLX-Gen
should fail early with a clear message. Silent or warning-only behavior is dangerous because the
output image can look plausible while ignoring the requested adapter.

This is now planned because the current behavior is inconsistent with ADR 0002 and the user-facing
troubleshooting docs: resolution is strict, but application can still degrade silently later.

## What we want to do

Add a capability matrix and strict LoRA application policy:

1. Add family-level capability metadata for LoRA inference and LoRA training.
2. Make the unified `mlxgen` router reject LoRA flags for unsupported families before model load.
3. Change LoRA loading so user-requested files must exist, load, and apply at least one mapped
   target.
4. Keep partial-match warnings for valid adapters, but fail zero-match adapters by default.
5. Add docs and generated capability metadata so AbstractVision can decide whether to display LoRA
   controls.

Initial support matrix should be explicit and task-aware:

| Family | Current MLX-Gen LoRA status | Task directions | Difficulty | Path |
| --- | --- | --- | --- | --- |
| FLUX.1 | Inference mapping exists. Fill, depth, control, redux, kontext, and in-context variants share the FLUX transformer loader path. | T2I, I2I, fill/inpaint/outpaint after item 0019 | Low for strictness; medium for variant-specific proof | Keep supported, fail zero-match adapters, validate at least one T2I row and one fill/edit row. Ensure required in-context LoRAs and user LoRAs are both explicit. |
| FLUX.2 Klein | Inference mapping exists; training adapter exists. | T2I, latent I2I, edit-reference, multi-reference | Low to medium | Keep supported, add strict loader tests and one visible T2I plus one I2I/edit validation row per representative package. |
| Qwen Image / Qwen Image Edit / Qwen Image Edit 2509 / 2511 | Inference mapping exists. Official Qwen Image Edit 2511 advertises integrated LoRA capability and many adapters exist in the HF model tree. | T2I where the model supports it; I2I edit-reference and multi-reference for edit models | Medium | Keep supported only after visible validation with real Qwen LoRAs. Validate base Qwen Image, Qwen Image Edit, 2509, and 2511 separately because prompt/image contracts differ. |
| Z-Image / Z-Image Turbo | Inference mapping and training adapter exist; one public LoRA slow test exists. | T2I, latent I2I where routed | Low to medium | Keep supported, make strict failure behavior common with FLUX/Qwen, and preserve visible LoRA regression output. |
| ERNIE Image / Turbo | Constructors accept `lora_paths` for compatibility but initializer ignores them. | T2I, latent I2I | Medium to high | Reject LoRA flags before model load until a real ERNIE mapping and a public adapter proof exist. Mapping may be feasible, but no current code applies it. |
| Bonsai | Initializer ignores LoRA; packed ternary/low-bit layout is not a normal adapter target. | T2I | High / not priority | Reject LoRA flags. Revisit only if Bonsai publishes adapter semantics that match the packed MLX runtime. |
| FIBO / FIBO Edit | LoRA is rejected today; no proven mapping. FIBO Edit is deprioritized and not a release-quality unified edit route. | T2I only for base FIBO; FIBO Edit disabled in unified generation | High / deferred | Keep rejected. Do not spend LoRA work here until base FIBO/FIBO Edit priority changes. |
| Wan2.2 TI2V/T2V/I2V | No mapping or constructor support. A14B has separate high-noise and low-noise transformers. | T2V, I2V | High | Track separately in [proposed item 0033](../proposed/0033_video_lora_for_t2v_i2v.md). Start with Wan only after item 0015 and video integrity work are stable. |
| SeedVR2 | No LoRA mapping; current route is restoration/upscale rather than generation. | Image restoration/upscale today; video restoration proposed in item 0032 | Low value / not priority | Reject LoRA flags. Treat model-specific restoration controls such as resolution and softness separately from LoRA. |

Task-direction roadmap:

| Direction | Near-term stance | Notes |
| --- | --- | --- |
| T2I | Support for FLUX.1, FLUX.2, Qwen Image, and Z-Image after strict application tests. | This is the easiest surface because one prompt produces one image and visible adapter effects are easy to compare. |
| I2I | Support only for families whose edit/latent route uses the same mapped transformer and has visual proof. | Validate latent I2I separately from edit-reference and multi-reference; a LoRA that works for T2I can still hurt edit adherence. |
| T2V | Proposed, not part of this planned item. | Requires Wan-specific mapping and temporal validation; see item 0033. |
| I2V | Proposed, not part of this planned item. | Same as T2V plus source-image identity and motion validation; see item 0033. |

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
- Docs and tests that align the public troubleshooting claim with runtime behavior.

## Non-goals

- Do not implement LoRA for every family.
- Do not implement Wan/video LoRA in this item; preserve that work in proposed item 0033 unless it
  is promoted.
- Do not add automatic LoRA downloads during generation.
- Do not change existing quantization policies.
- Do not bake LoRAs into prepared folders unless that behavior is explicitly documented and tested.

## Validation

- Unit test missing file, corrupt file, zero matched keys, partial matched keys, and successful
  application.
- Router tests proving LoRA accepted for Qwen/FLUX/Z-Image and rejected for ERNIE/Wan/Bonsai until
  implemented.
- Real-image contact sheet with Qwen-Image-Edit-2511 and at least one public 2511 LoRA.
- Real-image proof for each task direction marked supported:
  - T2I: one model-backed row per supported image family;
  - latent I2I: one source-preserving variation row where claimed;
  - edit-reference/multi-reference: one focused Qwen or FLUX.2 edit row where claimed;
  - canvas outpaint: after item 0019 route ownership is stable for the target model family;
  - native fill/inpaint outpaint: only after item 0019 or a follow-up item provides a validated
    fill/mask backend.
- Metadata test proving generated images preserve original LoRA paths and scales.
- Save/prepare test proving baked or stripped LoRA behavior is documented and deterministic.

## Progress checklist

- [x] Confirm unresolved LoRA paths now fail in `LoraResolution`.
- [x] Confirm loader-level missing/unreadable and zero-match cases can still avoid hard failure.
- [ ] Add explicit family-level LoRA capability metadata.
- [ ] Add task-direction LoRA metadata so UI/API callers know which modes can accept adapters.
- [ ] Reject LoRA flags for unsupported families before generation starts.
- [ ] Make loader-level missing, unreadable, corrupt, zero-match, and shape-invalid cases fail
      closed.
- [ ] Add focused tests for strict LoRA application.
- [ ] Update docs and generated capability metadata.

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
