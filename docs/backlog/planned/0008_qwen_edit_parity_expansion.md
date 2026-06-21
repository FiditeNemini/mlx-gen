# Planned: Qwen edit parity expansion

## Metadata

- Created: 2026-05-28
- Status: Planned
- Completed: N/A

## ADR status

- Governing ADRs: [ADR 0001](../../adr/0001_runtime_smoke_validation_for_model_routes.md), [ADR 0002](../../adr/0002_no_silent_automatic_fallbacks.md), [ADR 0003](../../adr/0003_runtime_truth_vs_consumer_convenience.md)
- ADR impact: None if Qwen control, masked edit, and structured conditioning remain task-specific
  capability metadata under the existing generation contract.

## Context

Qwen Image/Edit is currently the best strategic image lane for MLX-Gen: Apache 2.0, already ported,
already quantized with a working mixed q4/q8 policy, and directly relevant to AbstractVision image
editing. Online evidence as of 2026-05-28 makes Qwen-Image-Edit-2511 especially important: the
model card describes better consistency, lower drift, improved character preservation, industrial
design gains, geometric reasoning, and integrated LoRA capabilities.

Update 2026-06-05: standardized local I2I sequence validation in
[completed item 0025](../completed/0025_standardized_i2i_sequence_validation.md) found that
`AbstractFramework/qwen-image-edit-2511-4bit` passed only the cinematic edit row and failed complex
crash, pencil-crash, and composition rows. `AbstractFramework/qwen-image-edit-2511-8bit` passed
cinematic and crash rows, but only partially handled pencil-crash and failed to carry crash/debris
state into multi-reference composition. That makes Qwen edit parity a concrete follow-up rather
than a vague expansion idea.

Update 2026-06-11: this item should no longer sit in `proposed/`. The local Diffusers checkout has
real Qwen control and masked-edit pipelines, the official
[`Qwen/Qwen-Image-Edit-2509`](https://huggingface.co/Qwen/Qwen-Image-Edit-2509) card explicitly
advertises native ControlNet conditions such as depth, edge maps, keypoints, and sketches, and
public control weights now exist for current Qwen families. That makes structured Qwen control a
concrete missing capability inside an already-supported family rather than a speculative idea.

Update 2026-06-15: the official Diffusers Qwen docs now expose the missing surface clearly enough
that this is no longer just a parity curiosity. `QwenImageControlNetPipeline`,
`QwenImageEditPlusPipeline`, `QwenImageInpaintPipeline`, and `QwenImageEditInpaintPipeline` are
documented as first-class public routes.

Update 2026-06-15: the first narrow public masked-edit slice is now implemented. MLX-Gen exposes
masked edit on the Qwen edit route through `--mask-path`, `mlxgen capabilities` now surfaces a
distinct `qwen.inpaint` capability row, and `AbstractFramework/qwen-image-edit-2511-8bit` has an
accepted same-seed q8 proof row using the dedicated
`lightx2v/Qwen-Image-Edit-2511-Lightning` adapter. The published contact sheet covers two
masked-edit conditions: localized engine enhancement and localized crash repair.

Update 2026-06-15 (mask control follow-up): the masked-edit proof now also includes a no-mask
control using the same prompts, seeds, and Lightning adapter. Those control runs recompose the
full scene, which is the evidence that `--mask-path` is doing real localization work instead of
the model only following the instruction globally. The published control sheet now states the
important invariant explicitly: same `768x432` source image, same prompt, same seed, same
Lightning adapter, and only `--mask-path` changed.

Update 2026-06-15 (structured control): the first narrow public structured-control slice is now
implemented and proven. `mlxgen generate` accepts `--controlnet-image-path` on the exact
`AbstractFramework/qwen-image-8bit` public row, routes that request to `qwen.control`, and
injects the exact sidecar `InstantX/Qwen-Image-ControlNet-Union:diffusion_pytorch_model.safetensors`.
The accepted q8 proof uses the dedicated `lightx2v/Qwen-Image-Lightning` adapter on two same-seed
A/B rows: canny-guided pagoda and pose-guided portrait. The no-control columns keep the same
prompt, seed, and Lightning adapter; only the control image changes. The structured-control slice
is intentionally narrow: base q8 prepared package only, one control image, no edit-route control,
and no control-inpaint yet.

Update 2026-06-15 (next slice): the strongest next bounded expansion is now control-inpaint on the
base Qwen lane with `InstantX/Qwen-Image-ControlNet-Inpainting`. The user-facing request shape is
still "source image + mask + prompt", but the backend is different from the current Qwen edit
masked route: it adds a dedicated inpaint control branch on top of base Qwen instead of relying on
the edit model plus masked latent blending alone. The point of this slice is not a new kind of
prompt; it is stronger locality, tighter mask-boundary adherence, and fewer unintended whole-frame
changes on hard localized edits. That makes it a better next slice than broadening to multiple
edit-plus or control families at once.

Update 2026-06-16 (public explanation boundary): the current public docs now explicitly separate
three different Qwen concepts so users do not confuse them:

- current masked edit / inpaint (`qwen.inpaint`) on `AbstractFramework/qwen-image-edit-2511-8bit`;
- current structured control (`qwen.control`) on `AbstractFramework/qwen-image-8bit`;
- planned control-inpaint on base Qwen with `InstantX/Qwen-Image-ControlNet-Inpainting`.

The new guide and explainer are intentionally documentation-only for the planned route. They do not
claim support yet; they explain why control-inpaint would still feel like "edit one masked part of
an image" to the user, while using a stricter backend than the current edit-model mask route.
Those docs also now state the important implementation truth in plain language: ControlNet
inpainting is not a LoRA, not a replacement base model, and not universally "better". It is an
extra model package loaded alongside the base Qwen model when stronger locality and boundary
discipline matter more than minimal setup.

Update 2026-06-21 (base-Qwen control-inpaint): the next narrow slice is now implemented and
proven. `mlxgen generate` accepts `--image + --mask-path` on the exact
`AbstractFramework/qwen-image-8bit` q8 row, routes the request to `qwen.control-inpaint`, injects
the exact `InstantX/Qwen-Image-ControlNet-Inpainting:diffusion_pytorch_model.safetensors`
sidecar, and publishes a same-source same-mask same-seed Lightning proof bundle. The accepted
public sheet compares the new base-Qwen route against the existing Qwen Edit 2511 masked route on
localized engine enhancement and localized crash repair. The unified `mlxgen generate --help`
surface now also names `--mask-path` and `--controlnet-image-path` directly so the route is easier
to discover without backend-specific help.

Maintenance update on the accepted route:

- the exact public `guidance=1` Lightning proof path now skips inactive negative-prompt work,
  records only the effective negative-prompt metadata, and invalidates cached source/mask
  conditions when those files change in place;
- the accepted engine and repair proof rows are still visually stable, and the current M5 Max
  timings for the published rows are `17.29s` / `24.65s` and `17.74s` / `23.86s` generation / wall.

## Current code reality

- MLX-Gen has Qwen text-to-image and image-edit variants under `src/mflux/models/qwen/`.
- `ModelConfig` resolves Qwen Image, Qwen Image Edit 2509, and Qwen Image Edit 2511.
- `mlxgen generate` routes Qwen edit when the model is an edit model, the task is `edit`, or
  multiple input images are present without explicit img2img.
- The local Diffusers checkout includes:
  - `pipeline_qwenimage.py`
  - `pipeline_qwenimage_img2img.py`
  - `pipeline_qwenimage_edit.py`
  - `pipeline_qwenimage_edit_plus.py`
  - `pipeline_qwenimage_edit_inpaint.py`
  - `pipeline_qwenimage_inpaint.py`
  - `pipeline_qwenimage_layered.py`
  - `pipeline_qwenimage_controlnet.py`
  - `pipeline_qwenimage_controlnet_inpaint.py`
- The local Diffusers Qwen control pipelines already expose `control_image` inputs, multi-control
  batching, and dedicated inpaint/controlnet handling.
- Public upstream control weights now exist and are practical validation candidates:
  - `InstantX/Qwen-Image-ControlNet-Union`
  - `InstantX/Qwen-Image-ControlNet-Inpainting`
  - `alibaba-pai/Qwen-Image-2512-Fun-Controlnet-Union`
- MLX-Gen docs and model cards already emphasize Qwen mixed q4/q8 because full q4 quality was not
  good enough.
- MLX-Gen now exposes first-class Qwen mask inputs on the edit route through `--mask-path`.
- MLX-Gen now also exposes one exact structured-control route through `mlxgen generate`:
  `AbstractFramework/qwen-image-8bit` on `qwen.control` with `--controlnet-image-path`.
- The structured-control route is fail-closed at the public contract layer: the unified router
  injects the exact `InstantX/Qwen-Image-ControlNet-Union:diffusion_pytorch_model.safetensors`
  sidecar, and conflicting `--controlnet-model` values are rejected.
- MLX-Gen now also exposes one exact control-inpaint route through `mlxgen generate`:
  `AbstractFramework/qwen-image-8bit` on `qwen.control-inpaint` with `--image + --mask-path`.
- The control-inpaint route is also fail-closed at the public contract layer: the unified router
  injects the exact `InstantX/Qwen-Image-ControlNet-Inpainting:diffusion_pytorch_model.safetensors`
  sidecar, rejects conflicting `--controlnet-model` values, and keeps the generic user request
  shape instead of introducing a Qwen-only public flag.
- The public docs now include an explicit Qwen localized-editing route matrix covering the shipped
  `qwen.inpaint`, `qwen.control`, and `qwen.control-inpaint` rows, their exact request shapes, and
  their accepted proof sheets.
- Non-validated base-Qwen rows do not advertise `qwen.control`; the accepted public control slice
  is the prepared q8 route only.
- There is legacy mask/control plumbing in the inherited FLUX.1 command surface, but it is not
  wired into current unified Qwen routing.

## Problem

MLX-Gen currently has the core Qwen generation/edit path, but the broader Diffusers Qwen feature
surface is richer. The highest-value missing pieces are not random new models; they are adjacent
Qwen edit modes that users naturally expect once they can edit images:

- inpainting and masked edit;
- structured control on the exact prepared q8 base route;
- edit-plus / multi-image behavior;
- layered composition;
- broader structured control and ControlNet-inpaint with depth, edge, pose, sketch, and related
  condition maps after the first exact slice;
- LoRA validation for 2511 workflows.

## What we want to do

Make Qwen parity the next serious image-edit expansion after current LoRA truthfulness work:

1. Create a Qwen feature matrix comparing MLX-Gen to the local Diffusers Qwen pipelines.
2. Add tests that lock current Qwen route decisions: text-to-image, img2img, single-image edit,
   multi-image edit, and explicit task override.
3. Port one adjacent feature family at a time, starting with the Qwen route that has the cleanest
   public weights and strongest visible effect.
4. Keep Qwen-Image-Edit-2509 and Qwen-Image-Edit-2511 as the practical edit baselines for new
   Qwen control work, and use Qwen Image 2512 only where the public control weights clearly target
   that generation route.
5. Validate BF16/q8/mixed q4/q8 only for the exact rows that are actually claimed.

## Why

Qwen is permissively licensed and already works in MLX-Gen. Improving it likely gives more user
value per engineering hour than starting another large image model port. It also avoids depending
on non-commercial FLUX.1 Kontext for high-quality local editing.

## Requirements

- Keep Qwen structured control and masked edit fail-closed until exact model, route, and control
  weights are proven.
- Do not auto-generate control images silently. If MLX-Gen later offers helper generation for
  canny/depth/pose, it must be explicit in the CLI and metadata.
- Keep capability metadata honest: structured control, masked edit, and plain edit are not the
  same route even if they share some transformer weights.
- Prefer one public control family first, for example a union ControlNet or dedicated inpainting
  ControlNet, before adding multiple condition-specific routes.
- Preserve exact package identity. A request for a Qwen edit route with ControlNet must not fall
  back silently to latent img2img or a different Qwen family.
- Publish proof rows with contact sheets, prompts, control inputs, and command logs before
  claiming support in docs or capabilities.

## Suggested implementation

1. Write a Qwen capability matrix that separates:
   - plain text-to-image;
   - latent img2img;
   - single-image edit;
   - multi-reference edit;
   - masked edit/inpaint;
   - structured control;
   - structured control + inpaint.
2. Keep the first public structured-control slice narrow: one exact prepared q8 base row, one
   exact public sidecar package, one control image, and no edit/control mixing.
3. Reuse the existing Qwen transformer, tokenizer, scheduler, and LoRA strictness patterns before
   adding new abstractions.
4. Extend unified `mlxgen generate` only after the underlying Qwen route is proven. Capability
   output should expose the exact control/mask directions the selected row supports.
5. Add focused parity fixtures only where a math mismatch appears; do not start with large
   full-generation comparisons if the route wiring is obviously incomplete.

## Scope

- Qwen edit/control/inpaint parity for currently supported Qwen families.
- Structured-control route surfacing, mask/control inputs, and exact validation rows.
- Documentation and capability updates once routes are proven.

## Non-goals

- Do not port every Qwen pipeline in one pass.
- Do not make Qwen generation auto-download models, ControlNets, or LoRAs.
- Do not promote Qwen-Image-2.0 until public weights and Diffusers/Transformers loading paths are
  verified.
- Do not let Qwen-specific naming leak into the generic `mlxgen` contract unless the same concept
  is useful across families.
- Do not broaden the current public structured-control claim beyond the exact validated q8 row
  until another route has its own accepted proof artifacts.

## Dependencies and related tasks

- [0007 LoRA capability matrix and strict application](0007_lora_capability_matrix_and_strict_application.md)
- [0019 First-class I2I modes and outpaint/reframe UX](0019_first_class_i2i_modes_and_outpaint_reframe.md)
- `src/mflux/models/qwen/`
- `src/mflux/task_inference.py`
- `src/mflux/cli/mlx_gen.py`
- `/Users/albou/projects/gh/diffusers/src/diffusers/pipelines/qwenimage/`

## Expected outcomes

- A clear Qwen feature matrix with implemented vs unsupported rows.
- At least one validated masked-edit Qwen route with public proof artifacts.
- At least one validated structured-control Qwen route with public proof artifacts.
- At least one validated base-Qwen control-inpaint route with public proof artifacts.
- A public-facing explanation that distinguishes shipped masked edit, shipped structured control,
  and shipped base-Qwen control-inpaint without overstating broader support.
- Capability output and docs that tell AbstractVision exactly when Qwen can expose control/mask
  inputs.

## Validation

- Same prompt, seed, and dimensions across no-control vs controlled runs.
- Visible contact sheets for at least one structure-driven example and one masked-edit example.
- Exact model-handle or prepared-package proof rows only for the combinations actually tested.
- Focused `uv run pytest` coverage for route resolution, control/mask argument validation, and any
  parity or loader helpers added during implementation.

## Progress checklist

- [x] Write the Qwen feature matrix against local Diffusers pipelines.
- [x] Decide the first public Qwen control target and its proof weights.
- [x] Implement strict route selection and capability surfacing for that target.
- [x] Validate the first q8 masked-edit proof row with visible contact sheets and a no-mask control.
- [x] Validate the first q8 structured-control proof row with visible same-seed no-control vs control contact sheets.
- [x] Decide the next narrow Qwen parity slice after masked edit and structured control: base-Qwen control-inpaint with the exact InstantX inpainting sidecar.
- [x] Implement strict base-Qwen control-inpaint routing on the exact prepared q8 row.
- [x] Validate the first q8 base-Qwen control-inpaint proof row with visible contact sheets and accepted public artifacts.

## Guidance for the implementing agent

Favor narrow, reviewable Qwen parity PRs. Start with the strongest upstream evidence and one
public proof adapter family. If adding masks or structured controls requires new generic CLI/API
concepts, document the generic `mlxgen` contract first so it does not become Qwen-specific
vocabulary.

## Sources checked

- `src/mflux/models/qwen/`
- `src/mflux/cli/mlx_gen.py`
- Local Diffusers checkout Qwen pipelines under `diffusers/src/diffusers/pipelines/qwenimage/`
- Qwen-Image-Edit-2509 model card: https://huggingface.co/Qwen/Qwen-Image-Edit-2509
- Qwen-Image-Edit-2511 model card: https://huggingface.co/Qwen/Qwen-Image-Edit-2511
- Diffusers Qwen docs: https://huggingface.co/docs/diffusers/api/pipelines/qwenimage
- Diffusers modular quickstart: https://huggingface.co/docs/diffusers/modular_diffusers/quickstart
- InstantX Qwen control weights: https://huggingface.co/InstantX/Qwen-Image-ControlNet-Union
- InstantX Qwen inpainting ControlNet: https://huggingface.co/InstantX/Qwen-Image-ControlNet-Inpainting
- Alibaba PAI Qwen 2512 union ControlNet: https://huggingface.co/alibaba-pai/Qwen-Image-2512-Fun-Controlnet-Union
- Qwen-Image-2.0 technical report watch item: https://arxiv.org/abs/2605.10730
