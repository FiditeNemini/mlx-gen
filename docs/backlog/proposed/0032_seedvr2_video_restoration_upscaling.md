# Proposed: SeedVR2 video restoration and upscaling

## Metadata

- Created: 2026-06-07
- Status: Proposed
- Completed: N/A

## ADR status

- Governing ADRs: [ADR 0001](../../adr/0001_runtime_smoke_validation_for_model_routes.md), [ADR 0002](../../adr/0002_no_silent_automatic_fallbacks.md)
- ADR impact: May need a small ADR if MLX-Gen adds a general media-restoration task taxonomy
  beyond image upscale and video upscale.

## Context

SeedVR2 is officially a video restoration model. The ByteDance-Seed Hugging Face collection lists
`ByteDance-Seed/SeedVR2-3B` and `ByteDance-Seed/SeedVR2-7B` as video-to-video models, and the
SeedVR2 paper describes one-step video restoration. MLX-Gen 0.18.13 added direct official 3B/7B
checkpoint loading and reusable q8/q4 package preparation, but only for single-image restoration
and upscaling.

Sources checked:

- ByteDance-Seed SeedVR collection: https://huggingface.co/collections/ByteDance-Seed/seedvr
- `ByteDance-Seed/SeedVR2-3B`: https://huggingface.co/ByteDance-Seed/SeedVR2-3B
- `ByteDance-Seed/SeedVR2-7B`: https://huggingface.co/ByteDance-Seed/SeedVR2-7B
- Official SeedVR repository: https://github.com/ByteDance-Seed/SeedVR
- SeedVR2 paper: https://arxiv.org/abs/2506.05301

## Current code reality

- `mlxgen upscale` and `mflux-upscale-seedvr2` route to
  `src/mflux/models/seedvr2/cli/seedvr2_upscale.py`.
- `SeedVR2.generate_image(...)` returns `GeneratedImage` and accepts one image path, resolution,
  seed, and softness.
- The SeedVR2 model code includes 3D VAE and transformer components, but the public MLX-Gen
  pipeline currently prepares a single image, not a temporal clip.
- There is no `GeneratedVideo` output path, MP4 container handling, audio preservation, frame-rate
  metadata, temporal chunking, or video-specific progress surface for SeedVR2.
- Existing SeedVR2 q8/q4 package work is image-validation evidence only. It must not be reused as a
  video-support claim without a video-backed smoke.

## Problem or opportunity

Video restoration/upscaling would be valuable for AbstractVision and direct CLI users: low-quality
or compressed videos could be corrected, denoised, sharpened, and upscaled without converting each
frame manually. However, silently applying the image upscaler frame-by-frame would be the wrong
default because it can introduce temporal flicker and does not prove the official video model
contract.

## Proposed direction

Add a first-class SeedVR2 video restoration path only after porting the official video inference
contract:

1. Add a separate command surface such as:

   ```sh
   mlxgen upscale-video \
       --model ByteDance-Seed/SeedVR2-3B \
       --video-path input.mp4 \
       --resolution 720 \
       --softness 0.2 \
       --output restored.mp4
   ```

   A unified `mlxgen upscale --video-path ...` alias is acceptable if image and video inputs are
   mutually exclusive and errors are explicit.

2. Port official temporal inference semantics before optimization:
   - frame extraction with original FPS and frame count;
   - temporal window/chunk handling;
   - overlap or stitching behavior if required by the official implementation;
   - video-shaped latent packing;
   - output frame decode and MP4 encode;
   - optional audio copy-through when an input audio stream exists.

3. Keep a separate, explicit `--framewise` fallback only if users ask for it. Do not silently fall
   back from video restoration to independent image restoration.

4. Reuse current SeedVR2 model/package resolution and fail-closed package identity checks. Do not
   switch from a requested 7B package to a 3B package or from an HF handle to a local alias.

5. Add video-specific metadata:
   - source video path;
   - source and output frame size;
   - FPS, frame count, and duration;
   - model handle/path;
   - resolution/softness;
   - temporal chunk size and overlap;
   - whether audio was copied.

## Why it might matter

This would make MLX-Gen useful for cleanup and enhancement workflows, not only generation. It also
fits the official SeedVR2 model purpose better than image-only upscaling.

## Promotion criteria

Promote to `planned/` only when:

- the official SeedVR2 video inference code path has been audited against MLX-Gen's current
  image-only port;
- a minimal implementation plan exists for temporal clip packing and decode;
- one small local MP4 smoke is feasible on the target Apple Silicon machine;
- the desired command/API surface is clear enough for AbstractVision to expose without guessing.

## Validation ideas

- Unit tests for argument validation: image and video inputs are mutually exclusive; unsupported
  model/package requests fail before model load.
- Tiny MP4 smoke, for example 16 to 33 frames at a small resolution, preserving frame count and
  FPS.
- Contact sheet comparing source frames, official/source-model output frames, and q8/q4 package
  output frames if quantized packages are claimed.
- Temporal consistency review: sample adjacent frames for flicker or discontinuity.
- Metadata check for source/output video dimensions, FPS, frame count, audio-copy status, model,
  resolution, softness, and package identity.

## Non-goals

- Do not implement video generation. This is restoration/upscale only.
- Do not silently process video as independent images.
- Do not claim 3B/7B q8/q4 video readiness from existing image-only contact sheets.
- Do not add automatic downloads during restoration.

## Guidance for future agents

Start from the official video inference path and port it to MLX. Keep the first smoke small and
reproducible. If the temporal path is too expensive, keep this proposal open rather than shipping a
framewise approximation as the default.
