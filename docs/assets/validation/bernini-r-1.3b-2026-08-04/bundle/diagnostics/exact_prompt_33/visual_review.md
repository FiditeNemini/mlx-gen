# Exact-prompt 33-frame diagnostic review

Status: **FAIL**

This controlled diagnostic uses the upstream eight-reference prompt verbatim with the
Bernini-R 1.3B renderer: seed 8108, 320x192, 33 frames, 16 fps, 20 steps, BF16, and a
256 px condition-size cap. It is intentionally smaller and shorter than the release
profile, so it is diagnostic evidence rather than a release-quality run.

The prompt is 2,461 characters and tokenizes to 571 tokens. The runtime records
`prompt_truncated: true`, meaning the final 59 tokens are outside the 512-token
conditioning window. This is a faithfully recorded limitation, not a reason to accept
the output.

All 33 decoded frames were inspected at original detail in the two paged 5K MLX
timeline sheets. The output does not reproduce the requested multi-reference identity,
objects, or actions. Identity and geometry drift early; frames 25-32 show severe
mosaic-like corruption, anatomical collapse, and discontinuous color/shape changes.
The result therefore fails semantic fidelity and temporal quality.

Measured adjacent-frame diagnostics reinforce the visual finding:

- mean global MAE: 11.8724
- maximum global MAE: 52.1561
- maximum localized 32x32-tile MAE: 107.5143
- four-frame latent-boundary mean MAE: 19.3588
- non-boundary mean MAE: 9.3770
- boundary/non-boundary ratio: 2.0645

The whole-process Darwin peak physical footprint was 5,769,742,448 bytes. That proves
this diagnostic fits comfortably within an 18 GB machine; it does not prove visual
correctness.

Evidence:

- `r2v_eight_reference_exact_prompt_33f_20steps.mp4`
- `r2v_eight_reference_exact_prompt_33f_20steps.metadata.json`
- `mlx_contact_sheet_page_01.png` (frames 0-19)
- `mlx_contact_sheet_page_02.png` (frames 20-32)
- `mlx_worst_transitions_contact_sheet.png`
- `reference_contact_sheet.png`
- `official_contact_sheet_page_01.png` through `official_contact_sheet_page_05.png`

The upstream clip is a qualitative target only: its producing checkpoint and inference
recipe are not attested, so this review does not claim official Bernini-R 1.3B parity.
