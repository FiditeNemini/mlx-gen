# Generative Reframe Command Log

These commands generated the 2026-06-07 reframe proof assets.

## Source A: Isolated Object

```sh
uv run mlxgen generate \
  --model AbstractFramework/flux.2-klein-4b-8bit \
  --prompt "A clean product-style image of a small polished red rescue beacon, fully visible and centered on a flat snow field, plenty of empty pale snowy background around it, soft morning light, crisp details, wide composition" \
  --width 432 \
  --height 240 \
  --steps 12 \
  --seed 6101 \
  --metadata \
  --replace \
  --output validation_outputs/reframe_2026_06_07/source_a_isolated_beacon.png
```

## Source B: Cropped Starship

```sh
uv run mlxgen generate \
  --model AbstractFramework/flux.2-klein-4b-8bit \
  --prompt "A close-up cropped image of a futuristic silver starship stranded on a snowy alien plain, the camera is too close so the nose and one engine are cut off by the frame, icy cliffs in the background, crisp cinematic sci-fi details, daylight" \
  --width 432 \
  --height 240 \
  --steps 12 \
  --seed 6102 \
  --metadata \
  --replace \
  --output validation_outputs/reframe_2026_06_07/source_b_cropped_starship.png
```

## Reframe A: Extend Background Around A Fully Visible Object

```sh
uv run mlxgen generate \
  --model AbstractFramework/flux.2-klein-4b-8bit \
  --image validation_outputs/reframe_2026_06_07/source_a_isolated_beacon.png \
  --reframe-padding "15%,35%,15%,35%" \
  --prompt "Generatively reframe this image into a wider view. Keep the red rescue beacon fully visible near the center, preserve its shape and color, and extend the snowy field and soft pale background naturally on all sides. Do not crop the beacon. Do not add a second beacon." \
  --steps 16 \
  --seed 6111 \
  --metadata \
  --replace \
  --output validation_outputs/reframe_2026_06_07/flux2_reframe_a_background_extension.png
```

## Reframe B: Reveal A Full Object From A Close Crop

```sh
uv run mlxgen generate \
  --model AbstractFramework/flux.2-klein-4b-8bit \
  --image validation_outputs/reframe_2026_06_07/source_b_cropped_starship.png \
  --reframe-padding "25%,50%,25%,50%" \
  --prompt "Generatively reframe this close-up into a wider establishing shot. Reveal the entire futuristic silver starship in the snowy alien plain, including the nose, full hull, both engines, landing legs, surrounding snow, and icy cliffs. Keep the same starship identity and material. Do not crop the ship. Keep it sharp and centered in a coherent wide frame." \
  --steps 16 \
  --seed 6112 \
  --metadata \
  --replace \
  --output validation_outputs/reframe_2026_06_07/flux2_reframe_b_reveal_full_starship.png
```

## Qwen 2511 q8 Reframe A: Extend Background Around A Fully Visible Object

```sh
uv run mlxgen generate \
  --model AbstractFramework/qwen-image-edit-2511-8bit \
  --image validation_outputs/reframe_2026_06_07/source_a_isolated_beacon.png \
  --reframe-padding "15%,35%,15%,35%" \
  --prompt "Generatively reframe this image into a wider view. Keep the red rescue beacon fully visible near the center, preserve its shape and color, and extend the snowy field and soft pale background naturally on all sides. Do not crop the beacon. Do not add a second beacon." \
  --negative "close-up, cropped beacon, duplicated beacon, second beacon, blur, text, watermark" \
  --steps 20 \
  --guidance 4 \
  --seed 6121 \
  --metadata \
  --replace \
  --output validation_outputs/reframe_2026_06_07/qwen2511_q8_reframe_a_background_extension.png
```

## Qwen 2511 q8 Reframe B: Reveal A Full Object From A Close Crop

```sh
uv run mlxgen generate \
  --model AbstractFramework/qwen-image-edit-2511-8bit \
  --image validation_outputs/reframe_2026_06_07/source_b_cropped_starship.png \
  --reframe-padding "25%,50%,25%,50%" \
  --prompt "Generatively reframe this close-up into a wider establishing shot. Reveal the entire futuristic silver starship in the snowy alien plain, including the nose, full hull, both engines, landing legs, surrounding snow, and icy cliffs. Keep the same starship identity and material. Do not crop the ship. Keep it sharp and centered in a coherent wide frame." \
  --negative "close-up, cropped ship, cut off nose, cut off engines, duplicated spaceship, blurry ship, text, watermark" \
  --steps 20 \
  --guidance 4 \
  --seed 6122 \
  --metadata \
  --replace \
  --output validation_outputs/reframe_2026_06_07/qwen2511_q8_reframe_b_reveal_full_starship.png
```
