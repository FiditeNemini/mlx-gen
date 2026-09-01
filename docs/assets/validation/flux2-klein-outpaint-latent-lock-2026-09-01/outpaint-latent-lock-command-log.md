# FLUX.2 Klein Latent-Locked Outpaint Commands

These commands reproduce the 2026-09-01 strict-outpaint rows. Every row uses the same source
image, the same padding, the same seed and the same step count, so the recorded
`outpaint_source_restore_difference` values are directly comparable across weights.

Source: `docs/assets/validation/reframe-outpaint-2026-06-08/source-b-cropped-starship.png`
(432x240). The padding expands it to a 1040x272 canvas; `--outpaint-fill auto` resolves to `edge`
because the deepest padded side (right, 345px) stays inside the 384px edge-fill reach.

Guidance is omitted on purpose: the route takes the model's own default, which is 1.0 on
step-distilled Klein and 4.0 on base Klein.

## FLUX.2 Klein 4B / q8 (distilled)

```sh
mlxgen generate \
  --model AbstractFramework/flux.2-klein-4b-8bit \
  --image docs/assets/validation/reframe-outpaint-2026-06-08/source-b-cropped-starship.png \
  --outpaint-padding "5%,80%,5%,60%" \
  --prompt "Outpaint this close cropped starship image into a much wider realistic shot of the full spacecraft in the snowy canyon. Keep the existing compact silver spacecraft consistent, complete the missing nose, rounded hull, short tail, twin round rear engines, snow field, and ice cliffs in the newly added space. The entire ship must fit inside the final wide frame. No duplicated spacecraft, no repeated mountains, no text, no border." \
  --steps 16 \
  --guidance 1 \
  --seed 8512 \
  --metadata \
  --replace \
  --output flux2_klein_4b_q8_outpaint_b.png
```

## FLUX.2 Klein 9B / q8 (distilled)

```sh
mlxgen generate \
  --model AbstractFramework/flux.2-klein-9b-8bit \
  --image docs/assets/validation/reframe-outpaint-2026-06-08/source-b-cropped-starship.png \
  --outpaint-padding "5%,80%,5%,60%" \
  --prompt "Outpaint this close cropped starship image into a much wider realistic shot of the full spacecraft in the snowy canyon. Keep the existing compact silver spacecraft consistent, complete the missing nose, rounded hull, short tail, twin round rear engines, snow field, and ice cliffs in the newly added space. The entire ship must fit inside the final wide frame. No duplicated spacecraft, no repeated mountains, no text, no border." \
  --steps 16 \
  --guidance 1 \
  --seed 8512 \
  --metadata \
  --replace \
  --output flux2_klein_9b_q8_outpaint_b.png
```

## FLUX.2 Klein base 4B / q8 (seam control)

Same settings, base weights, so guidance defaults to 4.0 (true CFG).

```sh
mlxgen generate \
  --model AbstractFramework/flux.2-klein-base-4b-8bit \
  --image docs/assets/validation/reframe-outpaint-2026-06-08/source-b-cropped-starship.png \
  --outpaint-padding "5%,80%,5%,60%" \
  --prompt "Outpaint this close cropped starship image into a much wider realistic shot of the full spacecraft in the snowy canyon. Keep the existing compact silver spacecraft consistent, complete the missing nose, rounded hull, short tail, twin round rear engines, snow field, and ice cliffs in the newly added space. The entire ship must fit inside the final wide frame. No duplicated spacecraft, no repeated mountains, no text, no border." \
  --steps 16 \
  --seed 8512 \
  --metadata \
  --replace \
  --output flux2_klein_base_4b_q8_outpaint_b.png
```

## Recorded source-region drift

`outpaint_source_restore_difference` in each run's metadata sidecar: the mean absolute
difference (0-255) between the source crop and the same window of the generated canvas, on a
96px-wide resample. The route never post-blends, so this is what the latent lock alone holds.

| Model | Guidance | Drift | Wall time |
| --- | --- | --- | --- |
| `AbstractFramework/flux.2-klein-4b-8bit` | 1.0 | 6.01 | 55 s |
| `AbstractFramework/flux.2-klein-9b-8bit` | 1.0 | 3.72 | 81 s |
| `AbstractFramework/flux.2-klein-base-4b-8bit` | 4.0 | 5.69 | 100 s |

Wall times are end to end on an M5 Max, including the weight load; the 16 denoising steps
themselves are about 8 s on 4B.
