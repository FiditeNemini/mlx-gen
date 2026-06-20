# SeedVR2 video restoration command log

## Eiffel 1900 source 3B

```sh
uv run mlxgen upscale \
  --model ByteDance-Seed/SeedVR2-3B \
  --video-path '/Users/albou/Downloads/Panorama of the Eiffel Tower in 1900 Thomas Edison Vintage Video.mp4' \
  --start-seconds 16 \
  --max-frames 6 \
  --resolution 2x \
  --softness 0.0 \
  --metadata \
  --output validation_outputs/seedvr2_video_2026_06_19/eiffel_16s_6f_source3b_soft0.mp4
```

## Eiffel 1900 source 7B

```sh
uv run mlxgen upscale \
  --model ByteDance-Seed/SeedVR2-7B \
  --video-path '/Users/albou/Downloads/Panorama of the Eiffel Tower in 1900 Thomas Edison Vintage Video.mp4' \
  --start-seconds 16 \
  --max-frames 6 \
  --resolution 2x \
  --softness 0.0 \
  --metadata \
  --output validation_outputs/seedvr2_video_2026_06_19/eiffel_16s_6f_source7b_soft0.mp4
```

Both public proof runs preserved source FPS and wrote silent MP4 outputs with `audio_copied=false`.
The public docs use these two restored videos, the bounded source excerpt, the comparison MP4, and
the heuristic metrics JSON as the main proof bundle for SeedVR2 bounded video restoration.
