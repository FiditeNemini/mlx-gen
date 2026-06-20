# ADR 0004: SeedVR2 Video Host Safety And Proof Boundaries

Status: Accepted.

## Context

SeedVR2 video restoration can consume large amounts of unified memory on Apple Silicon, especially
when the request enlarges the source clip, keeps a model instance alive across several seeds, or
allows several restores to overlap on the same machine.

MLX-Gen also publishes proof assets for SeedVR2 video. Those proofs need two different roles kept
separate:

- route proof: the command ran, preserved FPS, wrote a valid MP4, and stayed inside the supported
  host-safety envelope;
- family-quality proof: a comparative claim that one checkpoint or profile is visually better than
  another.

Recent SeedVR2 video work showed that heuristic metrics and contact sheets are useful review aids,
but they are not enough to justify broad family-ranking claims on their own. Host safety and proof
scope both need durable policy.

## Decision

MLX-Gen treats SeedVR2 video restoration as a host-safety-sensitive runtime surface.

The public CLI safe profile for SeedVR2 video must:

- default video restore to a conservative profile aimed at source-size restoration;
- enable `--low-ram` automatically for video inputs;
- use sequential temporal chunking as the public CLI execution path instead of the bounded
  in-memory direct path;
- restore one SeedVR2 video job at a time through a runtime lock;
- rebuild the SeedVR2 model per video seed so transformer state is not retained across several
  video outputs in one process;
- fail closed when the requested video profile exceeds the conservative safe envelope;
- require an explicit `--force-unsafe-video-memory` override for profiles outside that envelope.

For SeedVR2 public proof:

- full restored MP4 files are the primary proof artifacts;
- timings, memory measurements, and heuristic metrics are supporting evidence only;
- heuristic scores must not be used as the sole basis for public 3B-versus-7B quality ranking;
- comparative family claims require direct visual proof and a clearly bounded, like-for-like
  validation profile.

## Consequences

### Positive

- Default SeedVR2 video commands favor host stability over aggressive enlargement experiments.
- Long-running desktop sessions are less likely to be disrupted by MLX-Gen video work.
- Public docs can state what is validated without implying a family ranking that the evidence does
  not support.

### Negative

- Some previously attempted video enlarge profiles are no longer part of the default safe path.
- Rebuilding the model per seed makes multi-seed video work slower.
- Users who want to experiment beyond the safe envelope must opt in explicitly.

### Neutral

- The internal bounded direct path may still exist for narrow programmatic use or future research,
  but it is not the default public CLI contract.
- Heuristic metrics remain useful for route-health checks such as drift, oversmoothing, and
  temporal instability.

## Enforcement

- SeedVR2 video CLI review must reject silent widening of the safe envelope.
- New SeedVR2 video profiles that enlarge the source or materially raise target area must fail
  closed by default unless they are placed behind explicit unsafe override.
- Public docs must describe the safe profile plainly and must not present heuristic-only family
  rankings as release truth.
- Backlog items or releases that claim new SeedVR2 video quality envelopes must link the exact
  command profile and preserved proof assets.

## Validation

Compliance is validated by:

- focused CLI tests for safe-profile rejection, auto low-RAM behavior, failure manifests, and
  fresh-model-per-seed execution;
- focused SeedVR2 video chunking and metadata tests;
- fast-suite coverage for the SeedVR2 CLI surface;
- preserved MP4 proof artifacts, command logs, and measured memory/timing data for any public video
  claim.

## Backlog links

- Route implementation: [0032 SeedVR2 video restoration and upscaling](../backlog/completed/0032_seedvr2_video_restoration_upscaling.md)
- Follow-up audio work: [0046 SeedVR2 video audio copy-through](../backlog/proposed/0046_seedvr2_video_audio_copythrough.md)

## Related

- [ADR 0001: Runtime Smoke Validation For Model Routes](0001_runtime_smoke_validation_for_model_routes.md)
- [ADR 0002: No Silent Automatic Fallbacks](0002_no_silent_automatic_fallbacks.md)
- `src/mflux/models/seedvr2/cli/seedvr2_upscale.py`
- `docs/upscaling.md`
