# Planned: Z-Image native inpaint

## Metadata

- Created: 2026-06-15
- Status: Planned
- Completed: N/A

## ADR status

- Governing ADRs: [ADR 0001](../../adr/0001_runtime_smoke_validation_for_model_routes.md), [ADR 0002](../../adr/0002_no_silent_automatic_fallbacks.md)
- ADR impact: None if native Z-Image inpaint remains a task-specific route under the existing
  generation contract. A small ADR may be warranted only if mask semantics become shared across
  several image families in one public contract.

## Context

Z-Image is already a real MLX-Gen family, but the public route surface is still incomplete. The
official Diffusers Z-Image docs document both `ZImageImg2ImgPipeline` and `ZImageInpaintPipeline`,
and the local upstream checkout also includes `pipeline_z_image_controlnet_inpaint.py`. That means
native mask-based inpaint is now an upstream capability gap inside an already-supported family.

## Current code reality

- MLX-Gen currently supports Z-Image and Z-Image-Turbo text generation plus latent img2img flows.
- `src/mflux/models/z_image/cli/z_image_generate.py` exposes text and latent image-to-image
  behavior, but not first-class mask-based inpaint.
- The local Diffusers checkout includes:
  - `pipeline_z_image_inpaint.py`
  - `pipeline_z_image_controlnet_inpaint.py`
- The official Z-Image docs show a native inpaint path for `Tongyi-MAI/Z-Image-Turbo`.
- Planned item 0019 already owns generic I2I/outpaint/reframe UX, but it does not cover a native
  Z-Image mask route today.

## Problem or opportunity

Users who already see Z-Image as a fast local image family will expect more than latent img2img.
Native inpaint is a clearer, safer, and more controllable edit primitive than asking users to bend
latent img2img into patch replacement.

## Proposed direction

Add a dedicated Z-Image native inpaint track only after picking one exact proof surface:

1. Start with `Tongyi-MAI/Z-Image-Turbo` native inpaint, not control-inpaint.
2. Define the minimal public contract explicitly:
   - source image input;
   - mask input;
   - prompt and negative prompt;
   - exact failure behavior when a mask is missing or malformed.
3. Keep route selection fail-closed. A request for Z-Image inpaint must not fall back silently to
   latent img2img.
4. Add capability surfacing that distinguishes:
   - Z-Image text generation;
   - Z-Image latent img2img;
   - Z-Image native inpaint.
5. Only after native inpaint is proven should MLX-Gen consider `controlnet_inpaint` as a separate
   follow-up.

## Why it might matter

This is probably the highest-value missing Z-Image feature that still fits MLX-Gen's current
product shape. It improves edit precision without introducing a whole new image family.

## Promotion criteria

This item is promoted because:

- `Tongyi-MAI/Z-Image-Turbo` is already a supported family in MLX-Gen and the official Diffusers
  docs expose a dedicated native inpaint route;
- the missing work is a bounded route expansion inside an existing family rather than a new
  architecture port;
- the public contract is straightforward: source image, mask image, prompt, negative prompt, and
  fail-closed route selection.

## Validation ideas

- Same source image, prompt, and seed across no-mask img2img vs native inpaint runs.
- Contact sheet showing source image, mask, and inpainted result.
- Exact metadata for source path, mask path, dimensions, seed, steps, model handle, and resolved
  route.
- Focused tests for mask argument validation and fail-closed route selection.

## Non-goals

- Do not bundle controlnet-inpaint into the first pass.
- Do not silently reuse latent img2img behavior as "inpaint."
- Do not claim support for unreleased Z-Image edit/omni models from this item alone.

## Guidance for future agents

Keep the first pass narrow and route-honest. If native inpaint lands cleanly, then decide whether
Z-Image control-inpaint deserves its own follow-up or belongs inside item 0019.

## Sources checked

- `src/mflux/models/z_image/cli/z_image_generate.py`
- Local Diffusers Z-Image pipelines under `/Users/albou/projects/gh/diffusers/src/diffusers/pipelines/z_image/`
- Diffusers Z-Image docs: https://huggingface.co/docs/diffusers/api/pipelines/z_image
- Z-Image-Turbo model card: https://huggingface.co/Tongyi-MAI/Z-Image-Turbo
