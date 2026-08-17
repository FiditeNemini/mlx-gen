# Proposed: SeedVR2 video decode artifact at larger latent geometries

## Metadata

- Created: 2026-08-17
- Status: Proposed

## ADR status

- Governing ADRs:
  - [ADR 0002](../../adr/0002_no_silent_automatic_fallbacks.md)
  - [ADR 0004](../../adr/0004_seedvr2_video_host_safety_and_proof_boundaries.md)
  - [ADR 0005](../../adr/0005_seedvr2_video_quality_proof_requires_five_second_reader_first_clips.md)
- ADR impact: ADR 0002 is directly engaged. A crash was replaced by silently corrupted output when
  the output-size bug below was fixed, and corrupt-but-plausible video is a worse failure than a
  refusal. Whether the route should refuse this regime is a product decision this item does not
  take.

## Context

Fixing the video output-size divergence (see "Related" below) made `480x360` sources reachable for
the first time: they previously died in `_streamed_video_restore` with

```
SeedVR2 streamed video noise slice shape mismatch. expected (1,16,31,44,60), got (1,16,31,44,58).
```

With the sizes reconciled the route runs to completion, but the restored video contains intermittent
corrupted frames. A reader describes them as several frames decomposed into one: the frame is built
from mismatched rectangular regions carrying content from different moments, and the defect repeats
so the clip flickers.

This was found while building a SwiftVR/SeedVR2 comparison, not by a synthetic probe.

## Evidence

Source: a 121-frame (5.04 s at 24 fps) excerpt of real archival footage, restored at `1x` with
`--temporal-chunk-size 29 --temporal-chunk-overlap 8`. Fidelity is the per-frame Pearson correlation
between the restored frame and its source frame; a corrupted frame decorrelates sharply.

| Build | Geometry | Profile | Frames below 0.90 |
| --- | --- | --- | ---: |
| HEAD (2cb6bf9) | 320x240 | safe | 0 / 121 |
| with the size fix | 320x240 | safe | 0 / 121 |
| with the size fix | 320x240 | `--force-unsafe-video-memory` | 0 / 121 |
| with the size fix | 384x288 | `--force-unsafe-video-memory` | 0 / 121 |
| with the size fix | 480x352 | `--force-unsafe-video-memory` | **16 / 121** |

The failing run has mean correlation 0.9396 against 0.9659 for SwiftVR on the same source, and a
minimum of 0.6935 on frame 0.

Three things are therefore ruled out:

- **Not the size fix.** At 320x240 the estimators agree before and after the change, and HEAD and the
  fixed build score identically (0.9602 vs 0.9606, both 0 bad frames).
- **Not the memory override.** `--force-unsafe-video-memory` at 320x240 is clean.
- **Not the window partitioner.** `WindowPartitioner` was probed directly for latent grids
  `(8,30,40)`, `(8,44,60)`, `(31,30,40)` and `(31,44,60)`, shifted and unshifted: every partition
  emits each index exactly once and `reverse(partition(x)) == x` holds exactly.

What is left is geometry. The bracket is clean at latent `40x30` and `48x36`, corrupt at `60x44`.

## The pattern points at causal-sliced decode

The corrupted frame indices are `0, 1, 5, 65, 69, 73, 77, 85, 89, 93, 97, 101, 105, 109, 113, 117`.
From frame 65 the spacing is exactly **4**, which is the latent temporal downscale factor: one pixel
frame in every latent group of four is wrong, and the rate worsens across the clip rather than being
tied to chunk boundaries (chunk stride here is 21, and the bad frames do not align to it).

`SeedVR2Initializer` sets `model.vae.set_causal_slicing(split_size=4)`
(`src/mflux/models/seedvr2/seedvr2_initializer.py:101`), which gives
`slicing_latent_min_size = 1`, so `_decode_with_slicing`
(`src/mflux/models/seedvr2/model/seedvr2_vae/vae.py:93-114`) walks the latent temporal axis one
latent frame at a time, carrying `CausalConv3d.memory` between slices.

Slice size was probed at 480x352:

| `split_size` | latent frames per slice | Frames below 0.90 |
| ---: | ---: | ---: |
| 4 (shipped) | 1 | 16 / 121 |
| 8 | 2 | **107 / 121** |
| 16 | 4 | OOM: single 79.4 GB buffer exceeds the 77.3 GB Metal limit |
| `None` | all | OOM: single 135.5 GB buffer |

Larger slices are dramatically worse, so `split_size=4` is the intended value and raising it is not a
fix. The decoder's causal memory semantics evidently depend on slice length, which is consistent with
the artifact living in `CausalConv3d.memory` handling rather than in the slicing arithmetic.

## Reproduction

```bash
ffmpeg -y -i <source> -vf scale=480:360 -frames:v 121 -c:v libx264 -crf 12 -pix_fmt yuv420p src.mp4
mlxgen upscale --model seedvr2-3b --video-path src.mp4 --resolution 1x \
  --temporal-chunk-size 29 --temporal-chunk-overlap 8 --force-unsafe-video-memory \
  --output out.mp4
```

Frame 0 of `out.mp4` shows the defect plainly. Scoring the clip against the source reproduces the
table above.

The override is required only because 480x352 needs about 70 GB for its minimum permitted 29-frame
chunk against a roughly 40 GB host-safe budget, and SeedVR2 refuses chunks below 29 frames. Measured
physical footprint stayed at 7.7 GB throughout, so the budget is gating on the MLX allocator
high-water mark rather than real memory pressure; that is worth revisiting separately.

## Proposed direction

1. Instrument `CausalConv3d.memory` across decode slices at a failing geometry and compare against a
   passing one. The period-4 signature says the fault is in what one slice hands the next.
2. Establish the real predicate. The bracket is only three points; find whether the boundary is a
   latent dimension, a window-count transition (`nh` goes from 2 to 3 between 48x36 and 60x44), or a
   memory-driven effect.
3. Decide the ADR 0002 question explicitly: until the artifact is fixed, should the route refuse
   geometries it cannot restore correctly? A refusal with a clear message is better than corrupt
   output, but the predicate must be real rather than a guessed pixel threshold, or it will block
   working configurations.
4. Only then consider whether a validated maximum geometry belongs in `docs/upscaling.md`.

## Non-goals

- Do not raise `split_size` to work around this; it makes the artifact far worse.
- Do not revert the output-size reconciliation. It is independently correct, it is what the VAE
  actually receives, and reverting it restores a hard crash rather than fixing the artifact.
- Do not add a resolution guard before the predicate is known.

## Related

- The output-size divergence this exposed, and its regression tests:
  `tests/seedvr2/test_video_output_size_consistency.py`.
- [0114](0114_swiftvr_streaming_restoration.md) - SwiftVR restored the same source at native
  480x360 with 0 of 121 frames below 0.90, which is what made the SeedVR2 defect visible by contrast.
