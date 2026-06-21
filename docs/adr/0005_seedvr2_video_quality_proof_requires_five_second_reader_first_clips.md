# ADR 0005: SeedVR2 Video Quality Proof Requires Five-Second Reader-First Clips

Status: Accepted.

## Context

Recent SeedVR2 video work produced several misleading proof shapes:

- sub-second clips could show that a command ran, but they did not reliably expose cadence errors,
  time-mixing, or motion distortion;
- sampled contact sheets and heuristic scores were useful review aids, but they were not enough to
  let a reader judge whether movement stayed coherent;
- terse labels such as `wavelet`, `lab`, and `off` were implementation names, not reader-first
  explanations.

For video restoration quality, the reader has to see motion. If the proof is shorter than a few
seconds, it is too easy to miss the exact failure modes that matter most: skipped-motion feel,
time dilation, repeated cadence artifacts, or frame-to-frame distortion.

## Decision

For SeedVR2 public video quality claims, MLX-Gen uses a reader-first minimum proof duration of
five contiguous seconds.

The rule is:

- any public quality claim about a SeedVR2 video profile, checkpoint, or comparison must use at
  least five seconds of contiguous source video at the original FPS;
- shorter clips are allowed only for narrow internal diagnostics and must not be presented as the
  primary quality proof;
- full-length video runs may still be kept as host-safety or long-run stability evidence, but they
  are secondary to the five-second reader-first proof when judging visible quality.

Required public proof artifacts for SeedVR2 video quality claims are:

- restored MP4 files for the compared candidates;
- one comparison MP4 when more than one candidate is being compared;
- one contiguous motion-strip sheet;
- at least one detail crop sheet focused on moving subjects or other distortion-prone regions;
- a readable validation report that explains what the labels mean, what profiles were rejected,
  and what conclusion the reader should draw.

Reader-facing labels must be intelligible. Raw implementation tokens may appear in parentheses, but
the visible description must explain the behavior in plain language.

## Consequences

### Positive

- Readers can judge motion quality directly instead of inferring from a sub-second clip or a single
  score.
- Public docs distinguish between route proof, long-run stress proof, and visible quality proof.
- SeedVR2 validation reports become easier to audit and harder to overclaim.

### Negative

- Public validation takes longer to produce than a tiny smoke clip.
- Some earlier proof artifacts become internal audit evidence instead of public-quality evidence.

### Neutral

- Tiny clips and sampled metrics still remain useful for fast local debugging.
- Long-run full-video runs remain valuable for memory, host-safety, and output-integrity checks.

## Enforcement

- Docs, backlog completion notes, and release evidence must not present sub-second SeedVR2 clips as
  the primary proof of video quality.
- New SeedVR2 comparison claims must link the exact five-second proof bundle and readable report.
- Review should reject terse labels when a reader cannot understand what a mode or profile does
  without reading code.
- Comparative family claims must be grounded in direct visual proof, not heuristic ranking alone.

## Validation

Compliance is validated by:

- preserved five-second proof MP4s and comparison MP4s;
- preserved motion-strip and detail-crop review sheets;
- a checked-in validation report with human-readable labels and conclusions;
- direct visual review of the preserved proof artifacts.

## Backlog links

- Route implementation: [0032 SeedVR2 video restoration and upscaling](../backlog/completed/0032_seedvr2_video_restoration_upscaling.md)
- Follow-up audio work: [0046 SeedVR2 video audio copy-through](../backlog/proposed/0046_seedvr2_video_audio_copythrough.md)

## Related

- [ADR 0004: SeedVR2 Video Host Safety And Proof Boundaries](0004_seedvr2_video_host_safety_and_proof_boundaries.md)
- `docs/upscaling.md`
- `docs/assets/validation/seedvr2-video-2026-06-21/`
