# Bernini-R 1.3B all-frame visual review

Verdict: **FAIL for required visual quality.** The runtime and media contracts execute, but none
of the five required quality cases clears the full-video bar.

Every MLX output frame was inspected through the MP4s and the 5K all-frame sheets. The bundle also
renders all 81 frames of each official comparison, the exact conditioned source frames, and five
highest-MAE before/after transition pairs per MLX clip. The structured record binds the reviewed
videos, every per-case sheet, the overview sheets, and the sheet manifest by SHA-256. This is an
engineering inspection, not an external human study.

## Required quality cases

- **R2V with eight references — fail.** The clip is nearly static, requested cup and music motion
  is absent, headphones and cup do not transfer visibly, and the face is blurred. Large changes
  occur at the temporal slice transitions 0→1, 4→5, 8→9, and 12→13.
- **RV2V garment edit — fail.** Pinstripes transfer partially, but motion is nearly static and
  frames 13–16 contain obvious block, doubling, and subject corruption.
- **RV2V pinstripe A/B — fail.** The reference affects the garment, but green sleeve artifacts,
  weak motion, and tail degradation prevent a quality pass.
- **RV2V black-shirt A/B — fail.** The black shirt and logo do not transfer; the floral garment
  remains, motion is weak, and the tail degrades.
- **V2V snowman insertion — fail.** A snowman appears, but the result is nearly static and frames
  13–16 contain severe cyan and block corruption.

## Diagnostic and structural cases

The no-reference, no-source, and no-reference A/B rows remain negative diagnostic evidence. The
848px and 1280px one-step rows are blurred structural smokes only. These rows prove execution or
help isolate input roles; they do not contribute to the visual-quality gate.

The all-frame A/B SSIM value of `0.841950` shows that the pinstripe and black-reference outputs
differ. It does not establish that either requested garment transfer succeeds.

See `visual_review.json` for exact hashes and case-level notes.
