# Completed: SeedVR2 video restoration and upscaling

## Metadata

- Created: 2026-06-07
- Status: Completed
- Completed: 2026-06-19

## ADR status

- Governing ADRs: [ADR 0001](../../adr/0001_runtime_smoke_validation_for_model_routes.md), [ADR 0002](../../adr/0002_no_silent_automatic_fallbacks.md)
- ADR impact: None. This item stays under the existing `mlxgen upscale` surface and keeps audio
  behavior explicit instead of introducing a new task taxonomy or silent fallback.

## Outcome

Completed as a full-video SeedVR2 restore v1 under `mlxgen upscale`.

MLX-Gen now supports:

- `--video-path` on the existing SeedVR2 upscale route;
- bounded clip restore with `--start-seconds` and `--max-frames`;
- full-video restore through sequential temporal chunking for longer clips;
- official temporal input handling instead of silent framewise fallback;
- source-FPS preservation by default;
- `GeneratedVideo` metadata for source clip geometry, timing, padding, and audio state;
- public proof for:
  - `ByteDance-Seed/SeedVR2-3B`
  - `ByteDance-Seed/SeedVR2-7B`
  - full-length Eiffel source-video proof on the official `3B` and `7B` routes
  - bounded q8 smoke on `AbstractFramework/seedvr2-3b-8bit`

Current explicit limitation:

- output is a silent MP4 even when the source clip contains audio; this is tracked separately in
  proposed item [0046](../proposed/0046_seedvr2_video_audio_copythrough.md).

## Related backlog items

- [0014 Shared progress callbacks for image and video pipelines](../completed/0014_shared_progress_callbacks.md)
- [0016 Wan video integrity release gate](../completed/0016_wan_video_integrity_release_gate.md)
- [0030 SeedVR2 upscale smoke, metadata, and quality defaults](../completed/0030_seedvr2_upscale_smoke_and_metadata.md)
- [0031 SeedVR2 official ByteDance checkpoint support](../completed/0031_seedvr2_official_bytedance_checkpoint_support.md)

## Context

SeedVR2 is officially a video restoration model. The ByteDance-Seed Hugging Face collection lists
`ByteDance-Seed/SeedVR2-3B` and `ByteDance-Seed/SeedVR2-7B` as video-to-video models, and the
SeedVR2 paper describes one-step video restoration. MLX-Gen 0.18.13 added direct official 3B/7B
checkpoint loading and reusable q8/q4 package preparation, but only for single-image restoration
and upscaling.

Update 2026-06-15: the published `ByteDance-Seed/SeedVR2-3B` route type is explicitly
`Video-to-Video`, so MLX-Gen's current image-only upscale path should now be treated as a useful
subset rather than an approximate match for the official model purpose.

Sources checked:

- ByteDance-Seed SeedVR collection: https://huggingface.co/collections/ByteDance-Seed/seedvr
- `ByteDance-Seed/SeedVR2-3B`: https://huggingface.co/ByteDance-Seed/SeedVR2-3B
- `ByteDance-Seed/SeedVR2-7B`: https://huggingface.co/ByteDance-Seed/SeedVR2-7B
- Official SeedVR repository: https://github.com/ByteDance-Seed/SeedVR
- SeedVR2 paper: https://arxiv.org/abs/2506.05301

## Closing code reality

- `mlxgen upscale` and `mflux-upscale-seedvr2` route to
  `src/mflux/models/seedvr2/cli/seedvr2_upscale.py`.
- `SeedVR2.generate_image(...)` still handles the original image-only path.
- `SeedVR2.generate_video(...)` still handles the bounded direct path and returns `GeneratedVideo`.
- `SeedVR2.restore_video_to_path(...)` now handles longer clips through sequential temporal
  chunks, overlap blending, and streamed MP4 output.
- `src/mflux/utils/video_util.py` now provides source inspection, bounded clip decode, frame-window
  decode, sequential overlapping window iteration, and writer cleanup helpers, all with source FPS,
  source dimensions, duration, frame count, and audio-presence metadata.
- Video save goes through the shared `GeneratedVideo`/MP4 path and existing video-health
  validation, even for chunked full-video restore.
- The public restore contract is explicit: preserve source FPS, trim temporary temporal padding
  back to the requested clip length, and record `audio_copied=false` when the input stream had
  audio but the current output path does not remux it.
- Existing q4/q7B SeedVR2 package work remains image-only evidence. This item closes on bounded
  official `3B` and `7B` source proof plus the earlier q8 3B package smoke.

## Problem

Video restoration/upscaling would be valuable for AbstractVision and direct CLI users: low-quality
or compressed videos could be corrected, denoised, sharpened, and upscaled without converting each
frame manually. Silently applying the image upscaler frame-by-frame would have been the wrong
default because it can introduce temporal flicker and does not prove the official video model
contract. The implementation had to follow the official temporal path closely enough to earn a real
video-support claim.

## What changed

- `mlxgen upscale` now accepts exactly one of `--image-path` or `--video-path`.
- SeedVR2 video restore gained bounded clip controls:
  - `--start-seconds`
  - `--max-frames`
- Longer video restore gained explicit chunk controls:
  - `--temporal-chunk-size`
  - `--temporal-chunk-overlap`
  - `--color-correction`
- The public video path now:
  - decodes a bounded source clip with original FPS;
  - preprocesses video frames as a clip, not as independent images;
  - pads the temporal length to the official `T == 1` or `(T - 1) % 4 == 0` rule for `sp_size=1`;
  - creates video-shaped condition tensors and noise latents;
  - trims decoded output back to the requested frame count;
  - writes `GeneratedVideo` metadata including source clip and audio state;
  - streams longer clips through sequential frame windows instead of keeping the full decoded
    source in memory at once;
  - cleans up partial output files on failure.
- SeedVR2 CLI now warns explicitly when a source video had audio and the restored output is silent.
- Video restore rejects `--vae-tiling`; the public low-memory path for long clips is
  `--low-ram --mlx-cache-limit-gb 8` plus temporal chunking.
- Public docs now cover image and full-video restore together and include model-backed video
  commands plus proof artifacts.

## Acceptance criteria

1. `mlxgen upscale` accepts exactly one of `--image-path` or `--video-path`.
2. Default video restore uses the official temporal path; any framewise mode is explicit.
3. Source frame count and FPS are preserved by default unless the user explicitly asks otherwise.
4. Video save goes through the existing `GeneratedVideo`/MP4 path and passes video-health
   validation.
5. Failures preserve enough context to debug model, seed, dimensions, FPS, frame count, and error
   phase.
6. Metadata records source video path, source/output dimensions, requested/final resolution, FPS,
   frame counts, duration, model identity, quantization, softness, temporal padding/chunking, and
   audio-copied state.
7. Audio behavior is explicit and never silently dropped.
8. One tiny official-source MP4 smoke is required before docs claim support.
9. If q8/q4 video support is claimed, preserve model-backed tiny video proof artifacts for those
   package rows.
10. Do not expand `mlxgen generate`, `mlxgen capabilities`, or the public generation-task resolver
    in this item.
11. Do not reuse existing image-only SeedVR2 evidence as video-readiness proof.

All criteria are met for the full-video v1 scope except audio copy-through, which is now explicit
and split into follow-up item 0046 instead of blocking completion of the restore path itself.

## Validation

- Focused tests:

```sh
uv run pytest \
  tests/arg_parser/test_seedvr2_upscale_argparser.py \
  tests/metadata/test_generated_video.py \
  tests/image_generation/test_seedvr2_upscale_metadata.py \
  -q
```

- CLI router checks:

```sh
uv run pytest \
  tests/cli/test_mlx_gen_router.py::test_upscale_subcommand_routes_to_seedvr2_command \
  tests/cli/test_mlx_gen_router.py::test_upscale_subcommand_does_not_enable_downloads \
  -q
```

- Fast suite:

```sh
make test-fast
```

- Official 3B full-video public proof:

```sh
uv run mlxgen upscale \
  --model ByteDance-Seed/SeedVR2-3B \
  --video-path '/Users/albou/Downloads/Panorama of the Eiffel Tower in 1900 Thomas Edison Vintage Video.mp4' \
  --resolution 1x \
  --softness 0.0 \
  --color-correction wavelet \
  --temporal-chunk-size 49 \
  --temporal-chunk-overlap 16 \
  --low-ram \
  --mlx-cache-limit-gb 8 \
  --metadata \
  --output validation_outputs/seedvr2_video_2026_06_19/eiffel_full_source3b_restore1x_lowram.mp4 \
  --replace
```

- Official 7B full-video public proof:

```sh
uv run mlxgen upscale \
  --model ByteDance-Seed/SeedVR2-7B \
  --video-path '/Users/albou/Downloads/Panorama of the Eiffel Tower in 1900 Thomas Edison Vintage Video.mp4' \
  --resolution 1x \
  --softness 0.0 \
  --color-correction wavelet \
  --temporal-chunk-size 49 \
  --temporal-chunk-overlap 16 \
  --low-ram \
  --mlx-cache-limit-gb 8 \
  --metadata \
  --output validation_outputs/seedvr2_video_2026_06_19/eiffel_full_source7b_restore1x_lowram.mp4 \
  --replace
```

Published proof assets:

- `docs/assets/validation/seedvr2-video-2026-06-19/eiffel_full_restore_3b_1x.mp4`
- `docs/assets/validation/seedvr2-video-2026-06-19/eiffel_full_restore_7b_1x.mp4`
- `docs/assets/validation/seedvr2-video-2026-06-19/eiffel_full_3b_7b_1x_contact_sheet.jpg`
- `docs/assets/validation/seedvr2-video-2026-06-19/eiffel_full_3b_7b_1x_metrics.json`
- `docs/assets/validation/seedvr2-video-2026-06-19/seedvr2_video_restore_full_command_log.md`
- `docs/assets/validation/seedvr2-video-2026-06-19/seedvr2_video_restore_full_stats_m5max.json`

Observed timings on `M5 Max`:

- official `ByteDance-Seed/SeedVR2-3B` full clip at `1x`: `1893.11s` generation, `1897.77s` wall
  time, peak MLX memory `39.37 GB`, max RSS `27.37 GB`
- official `ByteDance-Seed/SeedVR2-7B` full clip at `1x`: `1778.50s` generation, `1785.53s` wall
  time, peak MLX memory `53.50 GB`, max RSS `66.16 GB`

Both proof clips preserved source FPS and wrote `audio_copied=false`.

The public metric panel compares sampled restored frames after downscaling them back to the
original `320x240` source resolution so the numbers measure restoration behavior rather than raw
output size. On this exact Eiffel full clip:

- `3B 1x` scored `65.43` with `sharpness_gain 1.70x`, `contrast_gain 1.09x`,
  `temporal_ratio 1.13`, and `drift_mae 0.0688`;
- `7B 1x` scored `69.07` with `sharpness_gain 1.35x`, `contrast_gain 1.06x`,
  `temporal_ratio 1.07`, and `drift_mae 0.0587`.

That means the current regular `7B 1x` route was numerically steadier across the full archival
clip, while `3B 1x` stayed visually crisper and materially lower-memory. The higher `7B` heuristic
score should be read as lower-drift / smoother-temporal behavior, not as an unconditional visual
winner. The upstream 7B repository also includes a separate `7B-sharp` checkpoint, but this item
does not claim first-class MLX-Gen support for that variant.

Local limitation check:

- a bounded q8 run on `/Users/albou/Movies/SF.mp4` completed successfully but over-smoothed the
  already-clean modern source. The public docs now describe this as a fit limitation instead of a
  proof row.

## Why it mattered

This makes MLX-Gen useful for bounded local video cleanup and enhancement workflows, not only image
upscaling and generation. It also finally matches the official SeedVR2 model purpose with a real
video-backed support claim instead of an image-only approximation.

## Docs impact

Completed:

- `README.md`
- `docs/README.md`
- `docs/getting-started.md`
- `docs/api.md`
- `docs/upscaling.md`
- `docs/faq.md`
- `docs/architecture.md`
- `CHANGELOG.md`
- `llms.txt`
- `llms-full.txt`

## Non-goals that remain

- Do not treat this as video generation.
- Do not silently process video as independent images.
- Do not claim q4 or 7B video support from this item.
- Do not add automatic downloads during restoration.
- Do not expand `mlxgen generate`, `mlxgen capabilities`, or the public generation-task resolver
  in this item.

## Follow-ups

- [0046 SeedVR2 video audio copy-through](../proposed/0046_seedvr2_video_audio_copythrough.md)
- Broader q4 / 7B / longer-clip public proof should stay separate until those rows are actually
  generated and reviewed.
