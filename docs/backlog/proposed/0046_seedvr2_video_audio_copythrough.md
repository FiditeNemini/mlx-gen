# Proposed: SeedVR2 video audio copy-through

## Metadata

- Created: 2026-06-18
- Status: Proposed

## ADR status

- Governing ADRs: [ADR 0002](../../adr/0002_no_silent_automatic_fallbacks.md)
- ADR impact: None if MLX-Gen keeps the current restore path explicit about audio behavior and adds
  copy-through as an opt-in or clearly documented default remux step.

## Context

Completed item [0032](../completed/0032_seedvr2_video_restoration_upscaling.md) added bounded
SeedVR2 video restoration under `mlxgen upscale` and made audio behavior explicit:

- source FPS is preserved;
- restored output is currently a silent MP4;
- metadata records `audio_present` and `audio_copied=false`.

That is a legitimate bounded v1, but practical restoration workflows often need the original audio
to survive the round-trip.

## Problem or opportunity

Users restoring old or compressed clips usually want:

- restored video frames;
- original timing;
- original soundtrack when one exists.

The current contract is explicit and safe, but it still leaves users with a manual remux step.

## Proposed direction

Add explicit audio copy-through/remux support for SeedVR2 restored videos:

1. Keep the current video restore path as the source of truth for frames.
2. When the input clip has audio, remux the original audio stream onto the restored video output if
   frame count, FPS, and duration still align closely enough for a safe copy-through.
3. Record the outcome in metadata:
   - `audio_present`
   - `audio_copied`
   - optional `audio_copy_reason` when audio was present but not copied
4. Fail closed or warn explicitly when remux cannot be done safely; do not silently write a muted
   file while claiming audio preservation.

## Why it might matter

This is the smallest follow-up that makes SeedVR2 video restoration feel complete for normal user
workflows without changing the model path itself.

## Promotion criteria

Promote after:

- the bounded `0032` restore path is in place and documented; and
- one concrete remux strategy is chosen and validated on a tiny local clip with audio.
