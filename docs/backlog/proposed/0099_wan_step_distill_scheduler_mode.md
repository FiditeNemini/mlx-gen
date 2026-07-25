# Proposed: Wan step-distill scheduler mode (`denoising_step_list` parity)

## Metadata

- Created: 2026-07-25
- Status: Proposed (promotes the scheduler slice of [0041](0041_lightx2v_wan_distilled_model_loader_support.md))
- Completed: N/A
- Effort: S-M (scheduler mode + CLI surface + parity audit)

## ADR status

- Governing ADRs: [ADR 0001](../../adr/0001_runtime_smoke_validation_for_model_routes.md),
  [ADR 0002](../../adr/0002_no_silent_automatic_fallbacks.md) (an explicit
  step-list request must never silently fall back to the UniPC grid).
- ADR impact: None until shipped.

## Context (investigation provenance)

Item 0041's audit already established the gap: with `steps=4`,
`flow_shift=5.0`, `WanUniPCMultistepScheduler` lands on timesteps
`[999, 937, 833, 625]`, while the upstream LightX2V distill contract is the
explicit list `[1000, 750, 500, 250]` (`Wan22StepDistillScheduler`,
`denoising_step_list`, `boundary_step_index`, `enable_cfg=false`). This repo
cannot express an explicit step grid today.

The 2026-07-25 storyboard consistency investigation adds a consumer-side
reason to promote just this slice, independent of the native distilled-model
loader question (which stays in 0041):

- Chained storyboard scenes run the Lightning 4-step CFG-off recipe per hop;
  part of the observed per-hop stylization drift ("cartoonish" tone ramp,
  Hasler-Susstrunk colorfulness 32 → 41.5 across the real film) is plausibly
  the distilled CFG-off look compounding, but that attribution CANNOT be
  tested honestly while the sampler cannot reproduce the distill-native grid:
  every Lightning-vs-base comparison currently confounds recipe with grid.
- Prior v2v measurements in this repo's investigation history show CFG-on
  halves frame-to-frame drift (mean per-frame delta 17 vs 34 at matched
  strength) — sizing the true Lightning-vs-base trade for i2v chains needs
  the grid-exact distill mode first.

## What we want to do

1. Add an explicit step-grid mode to the Wan scheduler surface
   (`denoising_step_list` + `boundary_step_index`), fail-closed when combined
   with options it cannot honor (ADR 0002).
2. Parity audit: reproduce the upstream distill timesteps exactly (0041's
   validation idea, scoped to the scheduler only).
3. Re-run the Lightning 4/8-step vs base-recipe comparison on the preserved
   storyboard chain case with the grid held honest, and record the
   drift-vs-cost verdict (the investigation left this as the one un-run A/B —
   a 20-step CFG-on scene is ~10x a Lightning scene on the same host, which
   exceeded the investigation's GPU budget).

## Non-goals

- Native `lightx2v/Wan2.2-Distill-Models` checkpoint loading (stays in 0041;
  state-dict naming and package layout are a separate, larger problem).
- Changing the shipped Lightning LoRA profile defaults.

## Dependencies and related tasks

- [0041](0041_lightx2v_wan_distilled_model_loader_support.md) (parent audit;
  keep it for the loader/checkpoint question).
- BlackPixel companion track `proposed/storyboard_consistency_2026_07/`
  (consumes the drift-vs-cost verdict for its per-scene quality dial).

## Validation

- Scheduler unit parity vs the upstream distill timestep list.
- Fixed-seed chain A/B (Lightning grid-exact vs current UniPC approximation
  vs base 20-step CFG-on) with named metrics: Laplacian-variance sharpness,
  Hasler-Susstrunk colorfulness ramp, upper-band edge energy.

## Progress checklist

- [ ] Explicit step-grid scheduler mode, fail-closed combinations
- [ ] Upstream timestep parity audit
- [ ] Grid-honest Lightning vs base chain A/B with recorded verdict
