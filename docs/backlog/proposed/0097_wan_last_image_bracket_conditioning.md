# Proposed: Wan A14B i2v `last_image` bracket conditioning (first+last frame)

## Metadata

- Created: 2026-07-25
- Status: Proposed (storyboard cross-scene-consistency investigation, 2026-07-25)
- Completed: N/A
- Effort: S (code) + A/B gate (quality evidence)

## ADR status

- Governing ADRs: [ADR 0001](../../adr/0001_runtime_smoke_validation_for_model_routes.md)
  (the route needs real-checkpoint proof before any capability row ships),
  [ADR 0002](../../adr/0002_no_silent_automatic_fallbacks.md) (if A14B quality
  fails the gate, reject the flag on that route — no silent single-frame fallback).
- ADR impact: None until the route ships.

## Context (investigation provenance)

The 2026-07-25 three-agent storyboard consistency investigation (BlackPixel
backlog `proposed/storyboard_consistency_2026_07/`) traced why chained
storyboard scenes lose subject identity: each scene conditions on exactly ONE
VAE-encoded frame (the predecessor's last frame), which is systematically the
most degraded frame of a drifting clip. Measured on the real film chain:
predecessor-tail Laplacian-variance sharpness collapsed 952 → 497 → 93 across
the three handoffs, and a chain rerun reproduced the same collapse class
(30 at the S3→S4 handoff).

A verified porting fact makes a second anchor cheap: the local diffusers
reference (`diffusers/pipelines/wan/pipeline_wan_i2v.py`,
`prepare_latents`) implements `last_image` bracket conditioning in EXACTLY the
tensor layout this repo already ported for single-frame i2v:

- without `last_image`: `video_condition = cat([image, zeros(num_frames-1)])`
  — identical to `_build_first_frame_video_condition` in
  `src/mflux/models/wan/variants/wan2_2_ti2v.py`;
- with `last_image`: `video_condition = cat([image, zeros(num_frames-2),
  last_image])` and the latent mask keeps BOTH frame 0 and the final frame at
  1 before the same temporal-scale expansion this repo already performs in
  `_load_video_condition`.

The delta is ~30 lines in `_load_video_condition` (accept an optional second
image, pad with `num_frames-2` zeros, set the tail mask row) plus CLI/config
plumbing for a `--last-image` style flag.

## Why it might matter

For chained storyboards, bracket conditioning lets a host pin a scene's END
state (e.g. the next scene's intended start, or a clean subject reference),
turning open-loop drift into interpolation between two anchors. It is the
strongest identity lever that stays on the full-quality A14B path (unlike the
VACE 1.3B reference route, see 0100).

## Honesty constraints (quality gate)

- Official first+last-frame training exists for Wan 2.1 (`Wan2.1-FLF2V-14B`);
  Wan 2.2 A14B first+last quality is EMPIRICAL — community reports are
  positive but this repo must gate on its own A/B before advertising it.
- Gate: fixed-seed pairs (single-frame vs bracketed) on the preserved
  storyboard chain case; verdict on subject identity at the far end plus
  mid-clip motion plausibility (bracketing can produce a "morph" if the two
  anchors are inconsistent).
- If A14B fails the bar, reject `last_image` on 2.2 A14B routes with a clear
  error naming the evidence (ADR 0002), and record the verdict here.

## What we want to do

1. Port the bracket layout in `_load_video_condition` (optional last frame,
   mask tail row), mirroring the diffusers reference exactly.
2. Expose it on the Wan A14B i2v route (`--last-image <path>`), fail-closed on
   variants whose conditioning layout differs (TI2V-5B `expand_timesteps`
   path, VACE).
3. Run the fixed-seed A/B gate on the preserved storyboard case; keep the
   clips as the proof bundle.
4. Ship the capability row + docs only on a pass verdict.

## Non-goals

- Prefix-K-frames conditioning (K>1 leading frames): expressible in the same
  function but out-of-distribution for every Wan 2.x checkpoint this repo
  ships; SkyReels-V2-class territory. Not gated here; do not fold it in.
- Wan 2.1 FLF2V checkpoint integration (separate model family decision).

## Dependencies and related tasks

- BlackPixel companion: `proposed/storyboard_consistency_2026_07/` (handoff
  frame picker is the host-side half of the same failure).
- Related here: 0099 (step-grid scheduler parity for honest Lightning A/Bs),
  0100 (VACE identity-anchor recipe, the draft-tier alternative).

## Validation

- Layout parity: tensor-shape and mask-row assertions against the diffusers
  reference implementation on a tiny synthetic case.
- Quality: the fixed-seed A/B gate above, on real checkpoints (ADR 0001).

## Progress checklist

- [ ] Bracket layout ported behind an optional argument
- [ ] Route gating (A14B i2v only; fail-closed elsewhere)
- [ ] Fixed-seed A/B gate on the storyboard chain case
- [ ] Capability row + docs or a recorded rejection verdict
