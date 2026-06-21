# SeedVR2 Video Restoration Command Log

These are the exact bounded proof commands documented in `docs/upscaling.md`.

Set the source clip path once:

```sh
SOURCE_VIDEO="Panorama of the Eiffel Tower in 1900 Thomas Edison Vintage Video.mp4"
```

## 1x Safe Bounded Proof

### 3B

```sh
mlxgen upscale \
  --model ByteDance-Seed/SeedVR2-3B \
  --video-path "$SOURCE_VIDEO" \
  --start-seconds 70 \
  --max-frames 149 \
  --resolution 1x \
  --softness 0.0 \
  --color-correction wavelet \
  --temporal-chunk-size 29 \
  --temporal-chunk-overlap 8 \
  --low-ram \
  --mlx-cache-limit-gb 8 \
  --metadata \
  --output eiffel_70s_149f_3b_chunk29_overlap8_wavelet_1x_after_causal_slicing.mp4
```

### 7B

```sh
mlxgen upscale \
  --model ByteDance-Seed/SeedVR2-7B \
  --video-path "$SOURCE_VIDEO" \
  --start-seconds 70 \
  --max-frames 149 \
  --resolution 1x \
  --softness 0.0 \
  --color-correction wavelet \
  --temporal-chunk-size 29 \
  --temporal-chunk-overlap 8 \
  --low-ram \
  --mlx-cache-limit-gb 8 \
  --metadata \
  --output eiffel_70s_149f_7b_chunk29_overlap8_wavelet_1x_after_causal_slicing.mp4
```

## 2x Explicit Enlarged Proof

Validate the same clip at `1x` first. The enlarged `2x` proof is an explicit unsafe-memory run.

### 3B

```sh
mlxgen upscale \
  --model ByteDance-Seed/SeedVR2-3B \
  --video-path "$SOURCE_VIDEO" \
  --start-seconds 70 \
  --max-frames 149 \
  --resolution 2x \
  --softness 0.0 \
  --color-correction wavelet \
  --temporal-chunk-size 29 \
  --temporal-chunk-overlap 8 \
  --low-ram \
  --mlx-cache-limit-gb 8 \
  --force-unsafe-video-memory \
  --metadata \
  --output eiffel_70s_149f_3b_chunk29_overlap8_wavelet_2x_after_causal_slicing.mp4
```

### 7B

```sh
mlxgen upscale \
  --model ByteDance-Seed/SeedVR2-7B \
  --video-path "$SOURCE_VIDEO" \
  --start-seconds 70 \
  --max-frames 149 \
  --resolution 2x \
  --softness 0.0 \
  --color-correction wavelet \
  --temporal-chunk-size 29 \
  --temporal-chunk-overlap 8 \
  --low-ram \
  --mlx-cache-limit-gb 8 \
  --force-unsafe-video-memory \
  --metadata \
  --output eiffel_70s_149f_7b_chunk29_overlap8_wavelet_2x_after_causal_slicing.mp4
```
