# Outpaint Model Matrix Command Log

Every supported outpaint route, run on one source with one padding value.

- Source: `docs/assets/validation/reframe-outpaint-2026-06-08/source-b-cropped-starship.png` (432x240)
- Padding: `5%,80%,5%,60%` — expands to a 1040x272 canvas
- Conditioning canvas: `--outpaint-fill auto` resolved to `edge` on every row (the deepest side,
  345 px on the right, is inside the 384 px edge-fill reach)
- Machine: Apple M5 Max, 40-core GPU, 128 GB unified memory. Times are warm-cache; a first run
  after boot adds weight-load time.

FLUX.2 Klein prompt, used for all four Klein rows:

```
Outpaint this close cropped starship image into a much wider realistic shot of the full
spacecraft in the snowy canyon. Keep the existing compact silver spacecraft consistent, complete
the missing nose, rounded hull, short tail, twin round rear engines, snow field, and ice cliffs in
the newly added space. The entire ship must fit inside the final wide frame. No duplicated
spacecraft, no repeated mountains, no text, no border.
```

## FLUX.2 Klein 4B (distilled, q8) — 8.4 s

```sh
mlxgen generate \
  --model AbstractFramework/flux.2-klein-4b-8bit \
  --image docs/assets/validation/reframe-outpaint-2026-06-08/source-b-cropped-starship.png \
  --outpaint-padding "5%,80%,5%,60%" \
  --prompt "<FLUX.2 Klein prompt above>" \
  --steps 16 --guidance 1 --seed 8512 \
  --metadata --replace \
  --output klein_4b_q8_distilled.png
```

## FLUX.2 Klein 9B (distilled, q8) — 17.2 s

```sh
mlxgen generate \
  --model AbstractFramework/flux.2-klein-9b-8bit \
  --image docs/assets/validation/reframe-outpaint-2026-06-08/source-b-cropped-starship.png \
  --outpaint-padding "5%,80%,5%,60%" \
  --prompt "<FLUX.2 Klein prompt above>" \
  --steps 16 --guidance 1 --seed 8512 \
  --metadata --replace \
  --output klein_9b_q8_distilled.png
```

## FLUX.2 Klein Base 4B (q8) — 22.6 s

```sh
mlxgen generate \
  --model AbstractFramework/flux.2-klein-base-4b-8bit \
  --image docs/assets/validation/reframe-outpaint-2026-06-08/source-b-cropped-starship.png \
  --outpaint-padding "5%,80%,5%,60%" \
  --prompt "<FLUX.2 Klein prompt above>" \
  --steps 20 --guidance 4 --seed 8512 \
  --metadata --replace \
  --output klein_base_4b_q8.png
```

## FLUX.2 Klein Base 9B (q8) — 54.1 s

```sh
mlxgen generate \
  --model AbstractFramework/flux.2-klein-base-9b-8bit \
  --image docs/assets/validation/reframe-outpaint-2026-06-08/source-b-cropped-starship.png \
  --outpaint-padding "5%,80%,5%,60%" \
  --prompt "<FLUX.2 Klein prompt above>" \
  --steps 20 --guidance 4 --seed 8512 \
  --metadata --replace \
  --output klein_base_9b_q8.png
```

## Qwen Image Edit 2511 (q8) — 198.5 s

Qwen accepts a negative prompt and benefits from one on this source.

```sh
mlxgen generate \
  --model AbstractFramework/qwen-image-edit-2511-8bit \
  --image docs/assets/validation/reframe-outpaint-2026-06-08/source-b-cropped-starship.png \
  --outpaint-padding "5%,80%,5%,60%" \
  --prompt "Outpaint this close cropped image into a wider realistic snowy canyon shot while keeping the same compact pod-like silver starship design from the source. Complete the missing nose, rounded hull, short tail, twin round rear engines, snow field, and ice cliffs in the newly added space. The final ship must remain a compact rounded spacecraft, not an airplane, with no large wings. Preserve the same lighting and camera angle. No text, no frame, no border, no duplicate ship." \
  --negative "airplane, jet aircraft, long wing, black wing, flat wing, runway, text, border, frame, hard seam, split image, collage, duplicate spacecraft, duplicated mountains, repeated mountain peaks, distorted engines, melted hull, blurry ship, cropped ship, cut off hull, cut off engines" \
  --steps 20 --guidance 4 --seed 8413 \
  --metadata --replace \
  --output qwen_image_edit_2511_q8.png
```
