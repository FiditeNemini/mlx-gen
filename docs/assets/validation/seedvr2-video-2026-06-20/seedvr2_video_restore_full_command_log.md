# SeedVR2 full-video restoration command log

## Eiffel 1900 full clip source 3B

```sh
/usr/bin/time -l uv run mlxgen upscale \
  --model ByteDance-Seed/SeedVR2-3B \
  --video-path '/Users/albou/Downloads/Panorama of the Eiffel Tower in 1900 Thomas Edison Vintage Video.mp4' \
  --resolution 1x \
  --softness 0.0 \
  --color-correction wavelet \
  --metadata \
  --output validation_outputs/seedvr2_video_2026_06_20/eiffel_full_source3b_restore1x_current.mp4 \
  --replace
```

## Eiffel 1900 full clip source 7B

```sh
/usr/bin/time -l uv run mlxgen upscale \
  --model ByteDance-Seed/SeedVR2-7B \
  --video-path '/Users/albou/Downloads/Panorama of the Eiffel Tower in 1900 Thomas Edison Vintage Video.mp4' \
  --resolution 1x \
  --softness 0.0 \
  --color-correction wavelet \
  --temporal-chunk-size 5 \
  --temporal-chunk-overlap 1 \
  --force-unsafe-video-memory \
  --metadata \
  --output validation_outputs/seedvr2_video_2026_06_20/eiffel_full_source7b_restore1x_current.mp4 \
  --replace
```

## Notes

- Both proof runs were executed strictly sequentially.
- Both runs preserved the full source clip frame count and source FPS.
- Both runs wrote silent MP4 output with `audio_copied=false`.
- The `3B` full proof used the default safe streaming profile.
- The `7B` full proof used an explicit small-chunk streaming profile so the runtime per-chunk
  memory checks stayed below the host-safe budget on this machine.
