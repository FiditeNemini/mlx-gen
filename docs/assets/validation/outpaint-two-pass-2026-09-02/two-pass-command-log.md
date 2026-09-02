# Outpaint Two-Pass Command Log

Subject duplication on deep two-axis padding, and the single-axis split that removes it.

- Source: `docs/assets/validation/reframe-outpaint-2026-06-08/source-b-cropped-starship.png` (432x240)
- Machine: Apple M5 Max, 40-core GPU, 128 GB unified memory
- Prompt (deliberately names the subject, the worst case):

```
Outpaint this close cropped starship image into a wider realistic shot of the full spacecraft in the
snowy canyon. Complete the missing hull, engines, snow field, and ice cliffs in the newly added space.
```

## Corner-depth sweep (single pass)

`corner-sweep-9b-distilled.jpg`. Source at top-left, padding right and bottom by the same fraction
of the source width and height; rows 0.20, 0.25, 0.30, 0.40, 0.50, 0.60; columns seeds 8512, 99, 7.
Run on the released 0.32.1 single-pass route. Read by eye: 0.20-0.30 clean on every seed, 0.40
grows a second hull on one seed, 0.50 and 0.60 duplicate on two to three seeds each.

```sh
# ratio -> right,bottom: 0.20 -> 86,48  0.25 -> 108,60  0.30 -> 130,72
#                        0.40 -> 173,96  0.50 -> 216,120  0.60 -> 259,144
mlxgen generate --model AbstractFramework/flux.2-klein-9b-8bit \
  --image docs/assets/validation/reframe-outpaint-2026-06-08/source-b-cropped-starship.png \
  --outpaint-padding "0,259,144,0" \
  --prompt "<prompt above>" --steps 16 --guidance 1 --seed 8512 99 7 \
  --metadata --replace --output r0.60.png
```

## Two-pass split on the worst geometry

`two-pass-9b-distilled.jpg`. Row 1: source top-left, `0,256,148,0`, `--outpaint-passes auto`
(splits: `0,0,148,0` to 432x400, then `0,256,0,0` to 688x400). Row 2: the mirror, source
top-right, `0,0,148,256`. Row 3: the single-pass 0.60 sweep row for reference. Both split rows are
clean on all three seeds; the single-pass reference duplicates on two of three.

```sh
mlxgen generate --model AbstractFramework/flux.2-klein-9b-8bit \
  --image docs/assets/validation/reframe-outpaint-2026-06-08/source-b-cropped-starship.png \
  --outpaint-padding "0,256,148,0" \
  --prompt "<prompt above>" --steps 16 --guidance 1 --seed 8512 99 7 \
  --metadata --replace --output two_pass.png
# --outpaint-passes 1 reproduces the duplicating single-pass run and prints the corner warning.
```

Recorded `outpaint_source_restore_difference` on the split runs: 9.07 / 9.21 / 9.81 (top-left) and
9.11 / 7.91 / 8.59 (top-right), against 3.7-4.7 for one pass on this model: the source goes through
the latent lock and the VAE twice.

## Every outpaint route on the same geometry

`model-matrix.jpg`. Source top-left, `0,256,148,0`, `--outpaint-passes auto`, seeds 8512 and 99
(Qwen: seed 8512 only, single pass and split side by side). See the table in
`docs/reframe-outpaint.md`.

## Qwen Image Edit 2511: the source lock

`qwen-source-lock.jpg`. Same geometry and prompt (no `--negative`), seed 8512, 20 steps, guidance 4.

| Run | Source drift | Original restored |
| --- | --- | --- |
| source lock disabled, single pass (reference) | 68.3 | no (recomposed: ship shrunk and redrawn) |
| source lock disabled, two passes (reference) | 61.9 | no |
| source lock, two passes | 11.1 underneath (12.1 / 0.9 per pass) | yes, every pass |
| source lock, recorded envelope `5%,80%,5%,60%`, recorded prompt and negative, seed 8413 | 1.37 | yes |

```sh
mlxgen generate --model AbstractFramework/qwen-image-edit-2511-8bit \
  --image docs/assets/validation/reframe-outpaint-2026-06-08/source-b-cropped-starship.png \
  --outpaint-padding "0,256,148,0" \
  --prompt "<prompt above>" --steps 20 --guidance 4 --seed 8512 \
  --metadata --replace --output qwen_two_pass.png
```

The lock is a mask the run writes beside each canvas (`outpaint_canvas_mask.png`,
`outpaint_pass2_canvas_mask.png`): black over the source minus the 24 px transition band on the
sides that gained pixels, white over the new area, passed to the route's masked-edit input. The
restore threshold is 24.0 on every route: a locked window measures 1.4-15.6 from the source across
these runs, a recomposed one 62-68.

## FLUX.2: original crop restored, and a negative prompt on base weights

`base-negative-prompt.jpg`. Row 1: base 4B, single pass on the worst geometry, seeds 8512 / 99, no
negative prompt. Row 2: the same runs with
`--negative "a second spacecraft, duplicate ship, two ships, repeated hull, extra fuselage, text, border, frame"`.
Row 3: distilled 9B on the worst geometry with the automatic split (drift underneath 6.7, per pass
7.8 / 4.4), and base 4B on the recorded envelope (6.7), both with the original crop restored
(`outpaint_source_restore_applied: true`).

```sh
mlxgen generate --model AbstractFramework/flux.2-klein-base-4b-8bit \
  --image docs/assets/validation/reframe-outpaint-2026-06-08/source-b-cropped-starship.png \
  --outpaint-padding "0,256,148,0" --outpaint-passes 1 \
  --prompt "<prompt above>" \
  --negative "a second spacecraft, duplicate ship, two ships, repeated hull, extra fuselage, text, border, frame" \
  --steps 20 --guidance 4 --seed 8512 99 --metadata --replace --output base4b_negative.png
```
