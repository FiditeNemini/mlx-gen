# Planned: First-class I2I modes and outpaint/reframe UX

## Metadata
- Created: 2026-06-04
- Status: Planned
- Completed: N/A

## ADR status
- Governing ADRs: [ADR 0001](../../adr/0001_runtime_smoke_validation_for_model_routes.md), [ADR 0002](../../adr/0002_no_silent_automatic_fallbacks.md)
- ADR impact: May revise existing ADRs or add a small task-capability ADR before closure.

The durable policy to preserve is that public tasks should describe media direction
(`text-to-image`, `image-to-image`, `text-to-video`, `image-to-video`). Backend modes such as
latent img2img, edit-conditioned I2I, multi-reference I2I, masked fill, inpainting, and
outpainting should not be exposed as fake public tasks unless an ADR explicitly accepts that
taxonomy.

## Context

MLX-Gen now has public media-direction tasks, internal image-to-image modes, and a capability
planner from [completed item 0020](../completed/0020_generation_capability_contract.md). That
solves the task taxonomy problem for text/image/video routing, but the reframe/outpaint workflow is
still incomplete. Current code has FLUX.1 Fill support and lower-level canvas/mask utilities, but
users do not have a first-class `mlxgen generate` path that creates the expanded canvas and mask
for them. The FLUX README still references a missing `tools/create_outpaint_image_canvas_and_mask.py`
helper.

Standard terms:

- Inpainting: regenerate masked areas inside an existing canvas.
- Outpainting / image expansion: extend beyond the original image boundaries by placing the source
  on a larger canvas and masking the new border area.
- Generative fill / Fill: umbrella model or product term for masked filling, including inpainting
  and outpainting.
- Reframing: user-facing term for changing the crop or canvas. It is reliable outpainting only
  when backed by an expanded canvas plus mask.

## Current code reality

- `src/mflux/task_inference.py` exposes `GenerationCapability`, `ModelCapabilities`, and
  `GenerationPlan`. Resolver APIs return public media-direction tasks and separate internal modes
  such as `latent-img2img`, `edit-reference`, `multi-reference`, `text-video`, and
  `first-frame-i2v`.
- `src/mflux/cli/mlx_gen.py` routes through `GenerationPlan.handler_id`, exposes
  `mlxgen capabilities`, and keeps `--task edit` only as a compatibility alias for image-to-image
  edit/reference mode.
- `tests/cli/test_mlx_gen_router.py` now covers FLUX.2 default edit/reference I2I, explicit
  latent I2I with `--image-strength`, multi-reference I2I, and mode/option rejection.
- `src/mflux/models/common/latent_creator/latent_creator.py` implements latent img2img by resizing
  the input image to the target size, encoding it, blending it with noise, and denoising.
- `src/mflux/models/flux2/variants/edit/flux2_klein_edit.py`,
  `src/mflux/models/qwen/variants/edit/qwen_image_edit.py`, and
  `src/mflux/models/fibo/variants/edit/fibo_edit.py` implement distinct image-conditioned edit
  paths. These are still image-to-image at the public API level.
- `src/mflux/models/flux/variants/fill/flux_fill.py` and
  `src/mflux/models/flux/variants/fill/mask_util.py` implement the true masked fill path for
  FLUX.1 Fill.
- `src/mflux/utils/image_util.py` already has `expand_image(...)` and
  `create_outpaint_mask_image(...)`.
- `src/mflux/cli/parser/parsers.py` has an `add_image_outpaint_arguments(...)` helper, but the
  unified router does not expose a working outpaint/reframe flow.
- `src/mflux/models/flux/README.md` documents outpainting with a helper script that is not present
  in the repository.
- `docs/api.md`, `docs/faq.md`, `docs/getting-started.md`, `docs/troubleshooting.md`,
  `llms.txt`, and `llms-full.txt` now document the current split: latent img2img is for
  whole-image variation/restyle with `--image-strength`; edit/reference and multi-reference I2I do
  not use `--image-strength`; outpainting/reframing is not first-class in unified `mlxgen generate`
  yet.
- Qwen Image Edit, FLUX.2 Klein Edit, FIBO Edit, ERNIE I2I, Z-Image I2I, and FLUX.1 Kontext can
  attempt broader composition or resizing, but they do not lock the source pixels into an expanded
  masked canvas. They should not be documented as reliable outpainting unless a true mask/canvas
  contract is implemented and validated.

## Problem

Users still see multiple workflow words inside image-to-image:

- `image-to-image`
- `edit`
- `img2img`
- `reframe`
- `outpaint`

The public-task split is now clean, but reframe/outpaint still needs a first-class mode and command
surface. A request like "extend the image to the left" should not require users to manually create
an expanded canvas and mask before calling a lower-level fill script.

The outpaint path is also incomplete. The model and utilities exist, but the user must manually
assemble an expanded canvas and mask through a missing documented tool before calling
`mflux-generate-fill`.

## What we want to do

Make reframe/outpaint a first-class image-to-image workflow that creates the expanded canvas and
mask automatically, then runs FLUX.1 Fill through the capability/plan routing baseline from item
0020.

## Why

This improves user ergonomics and reduces false mental models. It also keeps routing aligned with
ADR 0002: MLX-Gen should infer only when the model/input contract is explicit and fail closed
otherwise. It should not silently swap model families or pretend that a resize-only I2I path is
equivalent to true outpainting.

## Requirements

- Extend the capability/plan contract with fill/outpaint mode only where a backend implements a
  real mask/canvas contract.
- Keep `--image-strength` latent-img2img-only; reject it for fill/outpaint modes unless a backend
  explicitly implements equivalent semantics.
- Add first-class outpaint/reframe UX for FLUX.1 Fill:
  - accept one source image;
  - accept CSS-style padding such as `0,25%,0,25%`;
  - create an expanded canvas with the source pasted at `(left, top)`;
  - create a binary mask where added regions are regenerated and the original image is preserved;
  - run `Flux1Fill.generate_image(...)`;
  - save useful metadata for source path, padding, expanded dimensions, mask mode, and fill model.
- Fail closed for non-fill outpaint requests unless a model has a proven true mask/canvas
  outpainting implementation.

## Suggested implementation

1. Extend `GenerationCapability`/`GenerationPlan` only as needed for `fill-outpaint` support.
2. Add a FLUX Fill reframe adapter:
   - CLI route through unified `mlxgen generate`;
   - Python helper such as `Flux1Fill.generate_outpaint(...)` or
     `Flux1Fill.reframe_image(...)`;
   - use `ImageUtil.expand_image(...)`, `ImageUtil.create_outpaint_mask_image(...)`, and
     `BoxValues` for canvas/mask construction.
3. Add `--outpaint-padding` as the preferred option and keep `--image-outpaint-padding` as a
   compatibility alias if practical.
4. Update completions, model cards, docs, and troubleshooting around outpaint/reframe.

## Scope

- First-class FLUX.1 Fill outpaint/reframe command/API.
- Tests and docs for outpaint canvas/mask behavior.

## Non-goals

- Do not implement Qwen inpaint/outpaint parity in this item. Keep that in
  [proposed item 0008](../proposed/0008_qwen_edit_parity_expansion.md) unless promoted.
- Do not claim Qwen, FLUX.2, FIBO, ERNIE, Z-Image, or Kontext are reliable outpainting paths
  unless they preserve an expanded canvas with a real mask contract and pass validation.
- Do not remove legacy model-specific CLIs in the same change.
- Do not silently choose another model family to satisfy outpaint.
- Do not run large image models just to validate routing; use a minimal model-backed smoke only
  where ADR 0001 requires it.

## Dependencies and related tasks

- [ADR 0001](../../adr/0001_runtime_smoke_validation_for_model_routes.md) for model-backed smoke
  validation before claiming a route works.
- [ADR 0002](../../adr/0002_no_silent_automatic_fallbacks.md) for fail-closed routing.
- [Completed item 0018](../completed/0018_taskless_generation_routing.md) for current taskless
  routing context.
- [Completed item 0020](../completed/0020_generation_capability_contract.md) for the public
  task/internal mode/capability planning baseline.
- [Proposed item 0008](../proposed/0008_qwen_edit_parity_expansion.md) for future Qwen
  inpainting/outpainting parity.
- `src/mflux/task_inference.py`
- `src/mflux/cli/mlx_gen.py`
- `src/mflux/models/flux/cli/flux_generate_fill.py`
- `src/mflux/models/flux/variants/fill/`
- `src/mflux/utils/image_util.py`
- `src/mflux/utils/box_values.py`

## Expected outcomes

- Reframe/outpaint has a single working command/API that prepares the canvas and mask internally.
- Docs accurately distinguish:
  - latent img2img;
  - edit-conditioned I2I;
  - multi-reference I2I;
  - inpainting;
  - outpainting / image expansion;
  - user-facing reframing.

## Current release boundary

As of the 0.18.10 release preparation, this item is intentionally still planned. The capability
contract and user documentation are in place, and non-fill models fail closed for outpaint padding.
The remaining work is the actual FLUX.1 Fill outpaint/reframe adapter, canvas/mask command/API,
tests, and one tiny model-backed smoke before any docs or model cards claim first-class
outpainting support.

## Validation

- Fast resolver tests:
  - fill-capable models expose `fill-outpaint` or equivalent internal mode only when supported;
  - non-fill models reject outpaint padding unless explicitly supported;
  - `--image-strength` is rejected for fill-outpaint modes.
- Fast router tests:
  - `dev-fill` or `black-forest-labs/FLUX.1-Fill-dev` plus outpaint padding routes to the fill
    outpaint adapter;
  - non-fill models reject outpaint padding unless explicitly supported.
- Utility tests:
  - percent and pixel padding create correct expanded canvas and mask;
  - no-op, negative, or malformed padding fails before model load;
  - original image region is black in the mask and generated border region is white.
- Model-backed smoke:
  - one tiny FLUX.1 Fill outpaint smoke with generated canvas/mask evidence before documenting the
    route as working.
- Docs checks:
  - README, `docs/api.md`, `docs/getting-started.md`, `docs/faq.md`, `docs/troubleshooting.md`,
    completions, and generated model-card examples teach outpaint/reframe as masked fill, not as
    generic latent I2I resizing.

## Progress checklist

- [ ] Add FLUX Fill outpaint/reframe adapter and Python helper.
- [ ] Add canvas/mask utility validation.
- [ ] Add CLI routing, parser, completion, and metadata support.
- [ ] Run focused tests.
- [ ] Run one tiny model-backed outpaint smoke.
- [ ] Update docs and generated model-card guidance.
- [ ] Decide whether ADR 0002 needs a task-taxonomy addendum before closure.

## Guidance for the implementing agent

Re-check the code first because routing has been changing quickly. Keep the public contract simple:
media direction is the task, implementation is a backend mode. Prefer explicit errors over silent
model swaps. Do not claim outpaint support for a model unless the implementation preserves an
expanded source canvas with a mask and the output has model-backed validation evidence.
