# Proposed: preview the x0 estimate instead of the noisy latent

## Metadata

- Created: 2026-08-14
- Status: Proposed
- Completed: N/A

## ADR status

- Governing ADRs: none directly.
- ADR impact: none; this changes what a preview shows, never what is saved.

## Context

Step-wise previews currently decode the in-flight latent `x_t` directly. Early in a run that
latent is dominated by noise, so the first previews show noise regardless of which decoder
renders them (measured 2026-08-14 on Z-Image Turbo, 8 steps: the tiny preview and the full-VAE
preview agree with each other at only 16 dB PSNR at step 0, because both are faithfully showing
a noisy latent; both reach 35 dB by the final step).

The reference TAESD previewing notebook does not decode `x_t`. It decodes the model's **x0
estimate** — the denoiser's current guess at the finished image — by hooking `scheduler.step`
and computing `pred_original_sample` before the step is applied. That is what makes previews
readable within the first couple of steps, which is the entire value of previewing a long run.

Our families are rectified-flow / flow-matching rather than DDPM, so the equivalent is:

```
x0_hat = x_t - sigma_t * v_pred
```

where `v_pred` is the transformer output for the step and `sigma_t` the current sigma. Both are
available at the scheduler-step boundary inside each family's denoise loop, but neither is
currently exposed to the callback surface (`call_in_loop` receives the post-step latent only).

## Why it matters

Early abort is the reason previews exist. On a Wan A14B shot or a large image generation, seeing
at step 2 that the composition, pose, or subject is wrong saves minutes per iteration. A preview
that shows noise until the run is nearly finished cannot serve that purpose.

## Proposed work

- Extend the in-loop callback contract so a family can supply its x0 estimate (or the `v_pred`
  and `sigma_t` needed to form it) alongside the post-step latent, without changing the default
  behavior for existing callbacks.
- Compute the estimate once per step and reuse it for both preview decode paths (tiny and full),
  so the comparison stays apples-to-apples.
- Keep decoding `x_t` available behind an option: it is the honest view of the actual sampler
  state and is useful when debugging schedulers.
- Re-run the preview consistency harness (`untracked/taesd_consistency/`) and report the
  early-step readability gain as the acceptance evidence: the step at which a preview first
  predicts the final composition should drop substantially.

## Risks

- The x0 estimate is itself unreliable at high sigma; previews will still change materially in
  the first steps and must not be presented as a commitment.
- Higher-order schedulers (UniPC, DPM-family correctors) do not have a single unambiguous
  `v_pred` per step; the contract must let a family decline to supply an estimate and fall back
  to `x_t` rather than fabricating one.
