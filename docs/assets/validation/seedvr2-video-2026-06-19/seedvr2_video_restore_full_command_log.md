# SeedVR2 full-video restoration command log

## Eiffel 1900 full clip source 3B

```sh
/usr/bin/time -l uv run mlxgen upscale \
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
  --replace \
  2>&1 | tee validation_outputs/seedvr2_video_2026_06_19/eiffel_full_source3b_restore1x_lowram.log
```

## Eiffel 1900 full clip source 7B

```sh
/usr/bin/time -l uv run mlxgen upscale \
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
  --replace \
  2>&1 | tee validation_outputs/seedvr2_video_2026_06_19/eiffel_full_source7b_restore1x_lowram.log
```

## Notes

- Both proof runs were executed strictly sequentially.
- Both runs preserved the full source clip frame count and source FPS.
- Both runs wrote silent MP4 output with `audio_copied=false`.
- For this archival source, the final public proof moved from `2x` bounded excerpts to `1x`
  native-resolution full-video restoration because `1x` produced the better full-clip balance of
  sharpness, drift, and temporal stability.
