# Completed: SeedVR2 video audio copy-through

## Metadata

- Created: 2026-06-18
- Status: Completed
- Completed: 2026-06-21

## ADR status

- Governing ADRs: [ADR 0002](../../adr/0002_no_silent_automatic_fallbacks.md)
- ADR impact: No new ADR was required. The shipped path stays explicit, fail-closed, and
  machine-readable.

## Outcome

Completed as the post-write audio contract for SeedVR2 video restore.

MLX-Gen now:

- preserves the matching source audio segment on SeedVR2 restored MP4s by default when source
  audio is present and the written output still aligns safely;
- keeps bounded and streamed SeedVR2 outputs on one shared post-write audio path;
- records `audio_present`, `audio_copied`, `audio_copy_mode`, and `audio_copy_reason` in restore
  metadata;
- fails the run instead of silently publishing a muted output when audio copy-through cannot be
  proven safe, unless the caller explicitly opts out with `--drop-audio`;
- publishes a real-source Air France `25s–35s` proof bundle for the shipped audio contract.

## Closing code reality

- `src/mflux/utils/video_util.py` now owns the shared audio-copy helper:
  `VideoUtil.copy_source_audio_to_video(...)`.
- The helper:
  - checks source audio presence;
  - validates restored MP4 duration before mux;
  - uses `ffmpeg` for sync-safe video-copy plus AAC audio mux;
  - validates the output container before replacing the original file;
  - returns a structured `AudioCopyResult`.
- `src/mflux/models/seedvr2/variants/upscale/seedvr2.py` now:
  - records actual bounded clip start timing;
  - calls the shared helper after video write and before final metadata export;
  - preserves source audio by default and requires `drop_audio=True` / `--drop-audio` for
    intentionally silent outputs.
- `src/mflux/models/seedvr2/cli/seedvr2_upscale.py` now tells users that audio copy-through is
  required by default rather than best-effort.
- `tests/utils/test_video_util_audio_copythrough.py` adds focused success and refusal coverage for
  the helper, and existing SeedVR2 metadata/chunking tests now lock the copied-vs-skipped
  contract.

## Validation

Focused automated validation:

- `tests/utils/test_video_util_audio_copythrough.py`
- `tests/seedvr2/test_seedvr2_video_chunking.py`
- `tests/image_generation/test_seedvr2_upscale_metadata.py`
- `tests/metadata/test_generated_video.py`

Accepted published proof bundle:

- [Air France source excerpt](../../assets/validation/seedvr2-video-audio-2026-06-21/air_france_25s_10s_source_excerpt.mp4)
- [Air France copied-audio output](../../assets/validation/seedvr2-video-audio-2026-06-21/air_france_25s_10s_audio_copied.mp4)
- [Air France stream report](../../assets/validation/seedvr2-video-audio-2026-06-21/air_france_25s_10s_stream_report.json)
- [Air France audio-copy report](../../assets/validation/seedvr2-video-audio-2026-06-21/air_france_25s_10s_audio_copythrough_report.md)
- [Air France command log](../../assets/validation/seedvr2-video-audio-2026-06-21/seedvr2_audio_copythrough_command_log.md)

Accepted result on the real `25.0s` to `35.0s` Air France window:

- FPS: `25.0`
- frame count: `250`
- video duration: `10.0s`
- audio duration: `10.0s`
- audio start time: `0.0s`
- output audio streams: `1`
- metadata outcome: `audio_copied=true`

The release contract now uses that refusal path to fail the run unless the caller explicitly opts
out with `--drop-audio`.

## Related backlog items

- [0032 SeedVR2 video restoration and upscaling](0032_seedvr2_video_restoration_upscaling.md)
- [0048 SeedVR2 enlarged-video safe-profile certification](../proposed/0048_seedvr2_enlarged_video_safe_profile_certification.md)

## Follow-ups

- Keep enlarged-profile certification separate in [0048](../proposed/0048_seedvr2_enlarged_video_safe_profile_certification.md).
- If future video families need audio support, reuse the same fail-closed output boundary only
  after they prove the abstraction is shared in practice.
