# SeedVR2 audio copy-through proof

## Scope

- Source clip: `Air France Commercial 2011 - LEnvol - Mozart K488 Adagio.mp4`
- Bounded window: `25.0s` to `35.0s`
- Purpose: validate the shipped post-write SeedVR2 audio copy-through contract on a real
  audio-bearing MP4 without publishing failed runtime experiments

## Accepted artifacts

- [source excerpt](./air_france_25s_10s_source_excerpt.mp4)
- [copied-audio output](./air_france_25s_10s_audio_copied.mp4)
- [stream report](./air_france_25s_10s_stream_report.json)
- [command log](./seedvr2_audio_copythrough_command_log.md)

## Result

The accepted output preserved the exact bounded video contract and copied audio successfully:

- video streams: `1`
- audio streams: `1`
- FPS: `25.0`
- frame count: `250`
- video duration: `10.0s`
- audio duration: `10.0s`
- audio start time: `0.0s`
- metadata outcome: `audio_copied=true`
- copy mode: `ffmpeg_copy_video_aac_audio`

This proves the shipped behavior:

1. SeedVR2 video restore preserves the matching source audio segment by default when the written
   MP4 duration still aligns safely.
2. The copied output stays CFR and preserves the bounded clip timing.
3. The audio path is fail-closed and machine-readable through `audio_copied` and
   `audio_copy_reason`.
4. When a caller intentionally wants a silent MP4, the explicit route is `--drop-audio`.

## Notes

- This proof validates the output contract at the exact post-write boundary used by SeedVR2 video
  restore.
- It is not a new visual-quality claim. Existing SeedVR2 contact sheets remain the quality proof
  surface.
