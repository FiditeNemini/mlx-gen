# Proposed: realtime video previews with tiny decoders (TAEHV family)

## Metadata

- Created: 2026-08-14
- Status: Proposed
- Completed: N/A

## ADR status

- Governing ADRs: none directly.
- ADR impact: none; preview-only surface, never on the save path.

## Goal

Let an application show a Wan, Qwen, or FIBO generation forming while it runs — a live clip or
frame updating each denoising step — at a cost that makes leaving it on reasonable. Tiny-decoder
previews shipped in 0.29.0 for the FLUX.1 and FLUX.2 image latent spaces; the video families are
the remaining half, and the one where the economics are strongest.

## Why video is the strongest case

Measured on this host (M5 Max, Wan 2.2 TI2V-5B q8, 256x256, 17 frames):

| | Image (FLUX.2 Klein, 512x512) | Video (Wan TI2V-5B, 17 frames) |
| --- | --- | --- |
| One denoising step | ~230 ms | ~340 ms |
| One full-VAE decode | 203 ms | 1.49 s |
| Decode / step | 0.9 | **4.4** |

An image decode costs about one denoising step; a video clip decode costs about four and a half.
Previewing every step through the full VAE would add roughly 440% to that run, so continuous
preview is impractical today on exactly the routes where runs last minutes and a user most wants
to see progress. Upstream reports the same ratio in memory terms: a full video VAE decode of 61
frames at 512x320 needs 6-9 GB against under 0.5 GB for TAEHV, so the tiny path is also what
makes previewing feasible on memory-constrained hosts.

## What needs building

The video checkpoints use **TAEHV**, a different architecture from the TAESD graph already
ported: NTCHW layout, causal-in-time `MemBlock`s that consume the previous timestep's activation,
and `TGrow`/`TPool` temporal resampling convolutions.

- Port `MemBlock`, `TGrow`, `TPool`, and the TAEHV decoder graph to MLX alongside the existing
  `TinyAutoencoder`, including `pixel_shuffle` postprocessing for the patch-2 checkpoints and the
  `frames_to_trim` leading-frame drop.
- Load the raw (non-diffusers) key layout these checkpoints publish, and their float16 storage.
- Extend the preview registry with the `wan.2.1`, `wan.2.2-ti2v`, and `qwen-image` latent spaces,
  each enabled only after an empirical check against that family's own VAE.
- Decide and document the preview unit for video: a single representative frame per step is the
  cheapest useful signal, while decoding the whole clip each step is closer to what a scrubbing
  UI wants. The current image decoders deliberately reject multi-frame latents rather than
  silently previewing frame 0, so this choice must be explicit.
- Note for implementers: Wan video decoding is lazy today — `generate_video` denoises and frames
  materialize at save time — so a preview path must decode independently of that generator rather
  than assuming decoded frames exist mid-run.
- Validate per family with the harness used for the image families: decode real in-flight latents
  with both decoders, report agreement, and include a negative control.

## Upstream mapping

VAE lineage was established by direct safetensors comparison, so the mapping does not need
re-deriving:

| Our family | latent channels | checkpoint |
| --- | --- | --- |
| qwen (Image, Edit, 2509, 2511) | 16 | `taew2_1` |
| wan 2.2 A14B (T2V/I2V), wan 2.1 VACE, bernini | 16 | `taew2_1` |
| wan 2.2 TI2V-5B | 48 | `taew2_2` |
| fibo (all variants) | 48 | `taew2_2` (unverified: repository is gated) |
| seedvr2 | 16 | none published |

Qwen-Image ships Wan 2.1's encoder verbatim with a retrained decoder, so `taew2_1` previews
approximate Wan's rendering of a Qwen latent. Expect a small systematic offset beyond ordinary
tiny-decoder error, and measure it before enabling that mapping.

## Relationship to 0112

Live progress display works without [0112](0112_x0_estimate_previews.md): a user watching a clip
form still gets useful feedback from mid-run frames. Early abort — stopping a rejected run before
paying for the remaining minutes — needs 0112, because previews currently decode the noisy latent
and are not judgeable in the first steps. Do 0112 first if the goal is saving time; do this item
first if the goal is a live preview surface for applications.

## Risks

- Single-frame image use of a video decoder is supported upstream but produces `t_upscale`
  frames that must be trimmed; getting the trim wrong silently shows the wrong frame.
- The Qwen decoder mismatch above means a preview can be self-consistent yet systematically off
  the final image; document it rather than tuning it away.
- Preview decoding allocates alongside generation. Wan runs are already the heaviest memory
  profile in the project, so the preview unit chosen above must be bounded rather than
  proportional to clip length.
