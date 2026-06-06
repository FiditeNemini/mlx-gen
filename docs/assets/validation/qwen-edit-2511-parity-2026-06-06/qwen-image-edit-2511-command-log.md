# Qwen Image Edit 2511 Parity Commands

These commands generated the 2026-06-06 Qwen Image Edit 2511 parity sheet. The shared source image
is `docs/assets/examples/spaceship-snow/01_t2i_spaceship_snow.png`. All runs use `432x240`,
40 steps, guidance `4`, and the unified image-to-image edit routes.

## Source Model

```sh
uv run mlxgen generate \
  --model Qwen/Qwen-Image-Edit-2511 \
  --image docs/assets/examples/spaceship-snow/01_t2i_spaceship_snow.png \
  --i2i-mode edit \
  --prompt "Convert the source image into a clean graphite pencil sketch on white paper. Preserve the same wide camera framing, the same spaceship shape, the icy canyon background, and the rear engines. Use thin gray pencil outlines with light hand shading only. The final image must clearly look like a hand drawn pencil sketch, not a blurred photo." \
  --width 432 \
  --height 240 \
  --steps 40 \
  --guidance 4 \
  --seed 9501 \
  --metadata \
  --replace \
  --output validation_outputs/qwen_edit2511_parity_2026_06_06/qwen2511_base_b_pencil.png
```

```sh
uv run mlxgen generate \
  --model Qwen/Qwen-Image-Edit-2511 \
  --image docs/assets/examples/spaceship-snow/01_t2i_spaceship_snow.png \
  --i2i-mode edit \
  --prompt "Edit the source into the same spaceship after a hard landing in the snow at dusk. Preserve the same wide camera angle, spaceship identity, rear engines, canyon cliffs, and framing. The ship must remain solid and sharp, but show a tilted hull, bent landing struts, broken ice chunks, disturbed snow, a shallow scrape trail, and a thin smoke plume. Use blue-hour dusk lighting. No blur, no mesh, no dissolve." \
  --width 432 \
  --height 240 \
  --steps 40 \
  --guidance 4 \
  --seed 9502 \
  --metadata \
  --replace \
  --output validation_outputs/qwen_edit2511_parity_2026_06_06/qwen2511_base_c_crash.png
```

```sh
uv run mlxgen generate \
  --model Qwen/Qwen-Image-Edit-2511 \
  --image validation_outputs/qwen_edit2511_parity_2026_06_06/qwen2511_base_b_pencil.png \
  --image validation_outputs/qwen_edit2511_parity_2026_06_06/qwen2511_base_c_crash.png \
  --i2i-mode multi-reference \
  --prompt "Use the first image as the graphite pencil sketch style reference and the second image as the hard-landing crash content reference. Produce one coherent wide image of the same spaceship crashed in the snowy canyon: graphite pencil outlines on white paper, visible tilted hull, disturbed snow, broken ice chunks, scrape trail, and a thin smoke plume. Preserve the spaceship identity and canyon layout. No blur, no colored photo, no text." \
  --width 432 \
  --height 240 \
  --steps 40 \
  --guidance 4 \
  --seed 9503 \
  --metadata \
  --replace \
  --output validation_outputs/qwen_edit2511_parity_2026_06_06/qwen2511_base_d_composition.png
```

## q8 Prepared Package

```sh
uv run mlxgen generate \
  --model AbstractFramework/qwen-image-edit-2511-8bit \
  --image docs/assets/examples/spaceship-snow/01_t2i_spaceship_snow.png \
  --i2i-mode edit \
  --prompt "Convert the source image into a clean graphite pencil sketch on white paper. Preserve the same wide camera framing, the same spaceship shape, the icy canyon background, and the rear engines. Use thin gray pencil outlines with light hand shading only. The final image must clearly look like a hand drawn pencil sketch, not a blurred photo." \
  --width 432 \
  --height 240 \
  --steps 40 \
  --guidance 4 \
  --seed 9501 \
  --metadata \
  --replace \
  --output validation_outputs/qwen_edit2511_parity_2026_06_06/qwen2511_q8_b_pencil.png
```

```sh
uv run mlxgen generate \
  --model AbstractFramework/qwen-image-edit-2511-8bit \
  --image docs/assets/examples/spaceship-snow/01_t2i_spaceship_snow.png \
  --i2i-mode edit \
  --prompt "Edit the source into the same spaceship after a hard landing in the snow at dusk. Preserve the same wide camera angle, spaceship identity, rear engines, canyon cliffs, and framing. The ship must remain solid and sharp, but show a tilted hull, bent landing struts, broken ice chunks, disturbed snow, a shallow scrape trail, and a thin smoke plume. Use blue-hour dusk lighting. No blur, no mesh, no dissolve." \
  --width 432 \
  --height 240 \
  --steps 40 \
  --guidance 4 \
  --seed 9502 \
  --metadata \
  --replace \
  --output validation_outputs/qwen_edit2511_parity_2026_06_06/qwen2511_q8_c_crash.png
```

```sh
uv run mlxgen generate \
  --model AbstractFramework/qwen-image-edit-2511-8bit \
  --image validation_outputs/qwen_edit2511_parity_2026_06_06/qwen2511_q8_b_pencil.png \
  --image validation_outputs/qwen_edit2511_parity_2026_06_06/qwen2511_q8_c_crash.png \
  --i2i-mode multi-reference \
  --prompt "Use the first image as the graphite pencil sketch style reference and the second image as the hard-landing crash content reference. Produce one coherent wide image of the same spaceship crashed in the snowy canyon: graphite pencil outlines on white paper, visible tilted hull, disturbed snow, broken ice chunks, scrape trail, and a thin smoke plume. Preserve the spaceship identity and canyon layout. No blur, no colored photo, no text." \
  --width 432 \
  --height 240 \
  --steps 40 \
  --guidance 4 \
  --seed 9503 \
  --metadata \
  --replace \
  --output validation_outputs/qwen_edit2511_parity_2026_06_06/qwen2511_q8_d_composition.png
```

## q4 Prepared Package

```sh
uv run mlxgen generate \
  --model AbstractFramework/qwen-image-edit-2511-4bit \
  --image docs/assets/examples/spaceship-snow/01_t2i_spaceship_snow.png \
  --i2i-mode edit \
  --prompt "Convert the source image into a clean graphite pencil sketch on white paper. Preserve the same wide camera framing, the same spaceship shape, the icy canyon background, and the rear engines. Use thin gray pencil outlines with light hand shading only. The final image must clearly look like a hand drawn pencil sketch, not a blurred photo." \
  --width 432 \
  --height 240 \
  --steps 40 \
  --guidance 4 \
  --seed 9501 \
  --metadata \
  --replace \
  --output validation_outputs/qwen_edit2511_parity_2026_06_06/qwen2511_q4_b_pencil.png
```

```sh
uv run mlxgen generate \
  --model AbstractFramework/qwen-image-edit-2511-4bit \
  --image docs/assets/examples/spaceship-snow/01_t2i_spaceship_snow.png \
  --i2i-mode edit \
  --prompt "Wide establishing shot of the same spaceship after a hard landing in the snow at dusk. Preserve the original wide camera angle, full spaceship fully visible inside the frame, rear engines visible, canyon cliffs visible on both left and right sides, and snowy ground foreground. Show a tilted hull, bent landing struts, broken ice chunks, disturbed snow, a shallow scrape trail, and a thin smoke plume. Use blue-hour dusk lighting. Keep the ship solid and sharp." \
  --negative "close-up, zoomed in, cropped spaceship, missing canyon cliffs, portrait crop, cut off engines, cut off hull, blur, mesh, dissolve, extra spaceship, text" \
  --width 432 \
  --height 240 \
  --steps 40 \
  --guidance 4 \
  --seed 9512 \
  --metadata \
  --replace \
  --output validation_outputs/qwen_edit2511_parity_2026_06_06/qwen2511_q4_c_crash_retry_wide.png
```

```sh
uv run mlxgen generate \
  --model AbstractFramework/qwen-image-edit-2511-4bit \
  --image validation_outputs/qwen_edit2511_parity_2026_06_06/qwen2511_q4_b_pencil.png \
  --image validation_outputs/qwen_edit2511_parity_2026_06_06/qwen2511_q4_c_crash_retry_wide.png \
  --i2i-mode multi-reference \
  --prompt "Use the first image as the graphite pencil sketch style reference and the second image as the hard-landing crash content reference. Produce one coherent wide image of the same spaceship crashed in the snowy canyon: graphite pencil outlines on white paper, visible tilted hull, disturbed snow, broken ice chunks, scrape trail, and a thin smoke plume. Preserve the spaceship identity, full wide framing, and canyon layout. No blur, no colored photo, no close-up, no cropped spaceship, no text." \
  --negative "colored photograph, blue photo, close-up, zoomed in, cropped spaceship, missing canyon cliffs, cut off engines, cut off hull, blur, mesh, dissolve, extra spaceship, text" \
  --width 432 \
  --height 240 \
  --steps 40 \
  --guidance 4 \
  --seed 9513 \
  --metadata \
  --replace \
  --output validation_outputs/qwen_edit2511_parity_2026_06_06/qwen2511_q4_d_composition.png
```
