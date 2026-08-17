# SwiftVR vs SeedVR2 3B/7B - 1x video restoration

Generated 2026-08-17 on an Apple M4 Max, 128 GB unified memory.

## What this proves and what it does not

This is an ADR 0005 reader-first proof: 121 contiguous frames, 5.04 seconds at the source's own
24 fps, restored at `1x` with no scaling. It compares **speed and stability** across three
restoration routes on one real degraded source, and it shows enough magnified detail for a reader to
judge the quality trade-off themselves.

It is **not** a general quality ranking. One clip of archival black-and-white film is a single point:
grain-heavy, monochrome, low-contrast, with crowd motion and static architecture. Nothing here
predicts behaviour on modern colour footage, faces at scale, or text.

## Source

A 121-frame excerpt of `paris-1.mp4`, early-1900s Paris street footage: heavy film grain, a moving
crowd under umbrellas, and a static ornamented facade. Downscaled to 384x288 before restoration -
see "Why 384x288" below.

## Results

| Candidate | Wall | FPS | Speed-up | Fidelity (mean) | Frames below 0.90 | Peak MLX | Physical |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| SwiftVR | 11.5 s | 10.52 | **41x** | 0.9588 | 0 / 121 | 11.1 GB | 8.5 GB |
| SeedVR2-3B | 471.0 s | 0.257 | 1.0x | 0.9647 | 0 / 121 | 57.9 GB | 7.8 GB |
| SeedVR2-7B | 458.8 s | 0.264 | 1.03x | **0.9809** | 0 / 121 | 67.8 GB | 17.6 GB |

Fidelity is the per-frame Pearson correlation between restored and source luma. It measures how much
the restoration departs from its input, so a *lower* number is not automatically worse - a restorer
is supposed to change the image - but a sharp drop indicates corruption, and a consistently low value
indicates aggressive reinterpretation.

**All three are temporally stable at this geometry**: no frame in any candidate falls below 0.90, and
the [motion strip](paris_121f_motion_strip.jpg) shows even cadence with no time-mixing across eight
consecutive frames.

**7B is marginally faster than 3B** (458.8 s vs 471.0 s), consistent with the existing 2x observation
in `docs/upscaling.md` that 7B is not uniformly slower than 3B.

## The quality trade-off, and a metric that lies about it

A naive sharpness proxy - mean absolute spatial gradient - ranks the candidates
SwiftVR 13.75, SeedVR2-3B 12.89, SeedVR2-7B 11.39 against a source of 9.52, which reads as "SwiftVR
recovers the most detail". **The detail crops show the opposite.**

In [the facade crop](paris_121f_detail_crop_facade.jpg), SwiftVR flattens the roof finials into
smooth blobs, erases the film grain entirely, and renders the window bays as posterised bands. The
high gradient energy comes from those hard synthetic edges, not from recovered structure.
SeedVR2-3B resolves the same finials as distinct sculptural forms, and SeedVR2-7B retains the most
natural film texture while still sharpening - it looks closest to a real film scan.

The [crowd crop](paris_121f_detail_crop_crowd.jpg) shows the same pattern on moving subjects: SwiftVR
produces a clean, plausible, noticeably synthetic image; the SeedVR2 pair keep more of what was
actually in the frame.

This is the concrete instance of a trade-off the SwiftVR paper's own numbers imply - it wins
no-reference perceptual metrics while losing full-reference ones - and it is exactly why ADR 0005
requires visible proof rather than a score. **Do not cite the gradient-energy column as a quality
ranking.**

## Why 384x288 rather than the native 480x360

SeedVR2 cannot currently restore this source at its native resolution.

- At 480x352 (the multiple-of-16 geometry SeedVR2 crops 480x360 to) the minimum permitted 29-frame
  chunk needs roughly 70 GB against a roughly 40 GB host-safe budget, and SeedVR2 refuses chunks
  below 29 frames. It therefore needs `--force-unsafe-video-memory`.
- With that override it runs, but produces intermittently corrupted frames: 16 of 121 below 0.90
  fidelity, one pixel frame corrupted in every latent group of four. See
  [seedvr2_3b_480x352_defect_frame0.png](seedvr2_3b_480x352_defect_frame0.png) and backlog item
  [0115](../../../backlog/proposed/0115_seedvr2_video_decode_artifact_at_larger_geometries.md).

384x288 is the largest geometry tested here where all three candidates are clean, so it is the
honest common ground for a comparison. SwiftVR itself has no such limit: it restored the full
573-frame `paris-1.mp4` at native 480x360 in 40.7 s at 11.3 GB peak, with audio preserved.

The SeedVR2 runs in this bundle still carry `--force-unsafe-video-memory` so that all three runs use
one geometry. The override was verified not to affect output: at 320x240, with and without it,
SeedVR2-3B scores 0 frames below 0.90 either way. Measured physical footprint stayed under 18 GB in
every run, so the host-safe budget is gating on the MLX allocator high-water mark rather than real
memory pressure.

## Reproduction

```bash
ffmpeg -y -i paris-1.mp4 -ss 5 -frames:v 121 -vf scale=384:288 \
  -c:v libx264 -crf 12 -pix_fmt yuv420p source384.mp4

mlxgen upscale --model swiftvr --video-path source384.mp4 --resolution 1x --output swiftvr.mp4

mlxgen upscale --model seedvr2-3b --video-path source384.mp4 --resolution 1x \
  --temporal-chunk-size 29 --temporal-chunk-overlap 8 --force-unsafe-video-memory --output s3b.mp4

mlxgen upscale --model seedvr2-7b --video-path source384.mp4 --resolution 1x \
  --temporal-chunk-size 29 --temporal-chunk-overlap 8 --force-unsafe-video-memory --output s7b.mp4
```

Numbers in `paris_121f_metrics.json`.

## Conclusion for a reader choosing a route

- Want it fast, or restoring long material offline? **SwiftVR**, at roughly 40x the throughput and a
  fifth of the memory, with the understanding that it reinterprets texture rather than recovering it.
- Want the most faithful result on grain-heavy archival material, and can afford minutes per second
  of video? **SeedVR2-7B**.
- **SeedVR2-3B** sits between the two on quality and offers no speed advantage over 7B here.
