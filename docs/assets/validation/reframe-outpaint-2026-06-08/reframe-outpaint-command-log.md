# Reframe And Outpaint Validation Commands

These commands reproduce the selected 2026-06-08 reframe/outpaint proof rows. The shared source image is `docs/assets/validation/reframe-outpaint-2026-06-08/source-b-cropped-starship.png`.

Each row uses one source image. `--reframe-padding` asks the edit model to create a wider generative view. `--outpaint-padding` builds an expanded canvas and uses adaptive source blending after generation when the source window still matches.

## Qwen Image Edit / source

### Reframe B

```sh
uv run mlxgen generate \
  --model Qwen/Qwen-Image-Edit \
  --image docs/assets/validation/reframe-outpaint-2026-06-08/source-b-cropped-starship.png \
  --reframe-padding "25%,50%,25%,50%" \
  --prompt "Generatively reframe this close-up into a wider establishing shot. Reveal the entire futuristic silver starship in the snowy alien plain, including the nose, full hull, both engines, landing legs, surrounding snow, and icy cliffs. Keep the same starship identity and material. Do not crop the ship. Keep it sharp and centered in a coherent wide frame." \
  --negative "close-up, cropped ship, cut off nose, cut off engines, duplicated spaceship, blurry ship, text, watermark" \
  --steps 20 \
  --guidance 4 \
  --seed 8201 \
  --metadata \
  --replace \
  --output validation_outputs/reframe_outpaint_matrix_2026_06_08/qwen_edit_source_reframe_b.png
```

### Outpaint B

```sh
uv run mlxgen generate \
  --model Qwen/Qwen-Image-Edit \
  --image docs/assets/validation/reframe-outpaint-2026-06-08/source-b-cropped-starship.png \
  --outpaint-padding "5%,80%,5%,60%" \
  --prompt "Outpaint this close cropped starship image into a much wider realistic shot of the full spacecraft in the snowy canyon. Keep the existing central spacecraft surface consistent, and complete the missing nose, full hull, tail, engines, snow field, and ice cliffs in the newly added space. The entire ship must fit inside the final wide frame with empty snow visible around it. Preserve the same lighting and camera angle. No text, no frame, no border, no duplicate ship." \
  --negative "text, border, frame, hard seam, split image, collage, duplicate spacecraft, duplicated mountains, repeated mountain peaks, distorted engines, melted hull, blurry ship, cropped ship, cut off hull, cut off engines" \
  --steps 20 \
  --guidance 4 \
  --seed 8212 \
  --metadata \
  --replace \
  --output validation_outputs/reframe_outpaint_matrix_2026_06_08/qwen_edit_source_outpaint_b_wide.png
```

## Qwen Image Edit / q8

### Reframe B

```sh
uv run mlxgen generate \
  --model AbstractFramework/qwen-image-edit-8bit \
  --image docs/assets/validation/reframe-outpaint-2026-06-08/source-b-cropped-starship.png \
  --reframe-padding "25%,50%,25%,50%" \
  --prompt "Generatively reframe this close-up into a wider establishing shot. Reveal the entire futuristic silver starship in the snowy alien plain, including the nose, full hull, both engines, landing legs, surrounding snow, and icy cliffs. Keep the same starship identity and material. Do not crop the ship. Keep it sharp and centered in a coherent wide frame." \
  --negative "close-up, cropped ship, cut off nose, cut off engines, duplicated spaceship, blurry ship, text, watermark" \
  --steps 20 \
  --guidance 4 \
  --seed 8201 \
  --metadata \
  --replace \
  --output validation_outputs/reframe_outpaint_matrix_2026_06_08/qwen_edit_q8_reframe_b.png
```

### Outpaint B

```sh
uv run mlxgen generate \
  --model AbstractFramework/qwen-image-edit-8bit \
  --image docs/assets/validation/reframe-outpaint-2026-06-08/source-b-cropped-starship.png \
  --outpaint-padding "5%,80%,5%,60%" \
  --prompt "Outpaint this close cropped starship image into a much wider realistic shot of the full spacecraft in the snowy canyon. Keep the existing central spacecraft surface consistent, and complete the missing nose, full hull, tail, engines, snow field, and ice cliffs in the newly added space. The entire ship must fit inside the final wide frame with empty snow visible around it. Preserve the same lighting and camera angle. No text, no frame, no border, no duplicate ship." \
  --negative "text, border, frame, hard seam, split image, collage, duplicate spacecraft, duplicated mountains, repeated mountain peaks, distorted engines, melted hull, blurry ship, cropped ship, cut off hull, cut off engines" \
  --steps 20 \
  --guidance 4 \
  --seed 8212 \
  --metadata \
  --replace \
  --output validation_outputs/reframe_outpaint_matrix_2026_06_08/qwen_edit_q8_outpaint_b.png
```

## Qwen Image Edit / q4

### Reframe B

```sh
uv run mlxgen generate \
  --model AbstractFramework/qwen-image-edit-4bit \
  --image docs/assets/validation/reframe-outpaint-2026-06-08/source-b-cropped-starship.png \
  --reframe-padding "25%,50%,25%,50%" \
  --prompt "Generatively reframe this close-up into a wider establishing shot. Reveal the entire futuristic silver starship in the snowy alien plain, including the nose, full hull, both engines, landing legs, surrounding snow, and icy cliffs. Keep the same starship identity and material. Do not crop the ship. Keep it sharp and centered in a coherent wide frame." \
  --negative "close-up, cropped ship, cut off nose, cut off engines, duplicated spaceship, blurry ship, text, watermark" \
  --steps 20 \
  --guidance 4 \
  --seed 8201 \
  --metadata \
  --replace \
  --output validation_outputs/reframe_outpaint_matrix_2026_06_08/qwen_edit_q4_reframe_b.png
```

### Outpaint B

```sh
uv run mlxgen generate \
  --model AbstractFramework/qwen-image-edit-4bit \
  --image docs/assets/validation/reframe-outpaint-2026-06-08/source-b-cropped-starship.png \
  --outpaint-padding "5%,80%,5%,60%" \
  --prompt "Outpaint this close cropped starship image into a much wider realistic shot of the full spacecraft in the snowy canyon. Keep the existing central spacecraft surface consistent, and complete the missing nose, full hull, tail, engines, snow field, and ice cliffs in the newly added space. The entire ship must fit inside the final wide frame with empty snow visible around it. Preserve the same lighting and camera angle. No text, no frame, no border, no duplicate ship." \
  --negative "text, border, frame, hard seam, split image, collage, duplicate spacecraft, duplicated mountains, repeated mountain peaks, distorted engines, melted hull, blurry ship, cropped ship, cut off hull, cut off engines" \
  --steps 20 \
  --guidance 4 \
  --seed 8212 \
  --metadata \
  --replace \
  --output validation_outputs/reframe_outpaint_matrix_2026_06_08/qwen_edit_q4_outpaint_b.png
```

## Qwen Image Edit 2509 / source

### Reframe B

```sh
uv run mlxgen generate \
  --model Qwen/Qwen-Image-Edit-2509 \
  --image docs/assets/validation/reframe-outpaint-2026-06-08/source-b-cropped-starship.png \
  --reframe-padding "25%,50%,25%,50%" \
  --prompt "Generatively reframe this close-up into a wider establishing shot. Reveal the entire futuristic silver starship in the snowy alien plain, including the nose, full hull, both engines, landing legs, surrounding snow, and icy cliffs. Keep the same starship identity and material. Do not crop the ship. Keep it sharp and centered in a coherent wide frame." \
  --negative "close-up, cropped ship, cut off nose, cut off engines, duplicated spaceship, blurry ship, text, watermark" \
  --steps 20 \
  --guidance 4 \
  --seed 8301 \
  --metadata \
  --replace \
  --output validation_outputs/reframe_outpaint_matrix_2026_06_08/qwen2509_source_reframe_b.png
```

### Outpaint B

```sh
uv run mlxgen generate \
  --model Qwen/Qwen-Image-Edit-2509 \
  --image docs/assets/validation/reframe-outpaint-2026-06-08/source-b-cropped-starship.png \
  --outpaint-padding "5%,80%,5%,60%" \
  --prompt "Outpaint this close cropped starship image into a much wider realistic shot of the full spacecraft in the snowy canyon. Keep the existing central spacecraft surface consistent, and complete the missing nose, full hull, tail, engines, snow field, and ice cliffs in the newly added space. The entire ship must fit inside the final wide frame with empty snow visible around it. Preserve the same lighting and camera angle. No text, no frame, no border, no duplicate ship." \
  --negative "text, border, frame, hard seam, split image, collage, duplicate spacecraft, duplicated mountains, repeated mountain peaks, distorted engines, melted hull, blurry ship, cropped ship, cut off hull, cut off engines" \
  --steps 20 \
  --guidance 4 \
  --seed 8312 \
  --metadata \
  --replace \
  --output validation_outputs/reframe_outpaint_matrix_2026_06_08/qwen2509_source_outpaint_b.png
```

## Qwen Image Edit 2509 / q8

### Reframe B

```sh
uv run mlxgen generate \
  --model AbstractFramework/qwen-image-edit-2509-8bit \
  --image docs/assets/validation/reframe-outpaint-2026-06-08/source-b-cropped-starship.png \
  --reframe-padding "25%,50%,25%,50%" \
  --prompt "Generatively reframe this close-up into a wider establishing shot. Reveal the entire futuristic silver starship in the snowy alien plain, including the nose, full hull, both engines, landing legs, surrounding snow, and icy cliffs. Keep the same starship identity and material. Do not crop the ship. Keep it sharp and centered in a coherent wide frame." \
  --negative "close-up, cropped ship, cut off nose, cut off engines, duplicated spaceship, blurry ship, text, watermark" \
  --steps 20 \
  --guidance 4 \
  --seed 8301 \
  --metadata \
  --replace \
  --output validation_outputs/reframe_outpaint_matrix_2026_06_08/qwen2509_q8_reframe_b.png
```

### Outpaint B

```sh
uv run mlxgen generate \
  --model AbstractFramework/qwen-image-edit-2509-8bit \
  --image docs/assets/validation/reframe-outpaint-2026-06-08/source-b-cropped-starship.png \
  --outpaint-padding "5%,80%,5%,60%" \
  --prompt "Outpaint this close cropped starship image into a much wider realistic shot of the full spacecraft in the snowy canyon. Keep the existing central spacecraft surface consistent, and complete the missing nose, full hull, tail, engines, snow field, and ice cliffs in the newly added space. The entire ship must fit inside the final wide frame with empty snow visible around it. Preserve the same lighting and camera angle. No text, no frame, no border, no duplicate ship." \
  --negative "text, border, frame, hard seam, split image, collage, duplicate spacecraft, duplicated mountains, repeated mountain peaks, distorted engines, melted hull, blurry ship, cropped ship, cut off hull, cut off engines" \
  --steps 20 \
  --guidance 4 \
  --seed 8312 \
  --metadata \
  --replace \
  --output validation_outputs/reframe_outpaint_matrix_2026_06_08/qwen2509_q8_outpaint_b.png
```

## Qwen Image Edit 2509 / q4

### Reframe B

```sh
uv run mlxgen generate \
  --model AbstractFramework/qwen-image-edit-2509-4bit \
  --image docs/assets/validation/reframe-outpaint-2026-06-08/source-b-cropped-starship.png \
  --reframe-padding "25%,50%,25%,50%" \
  --prompt "Generatively reframe this close-up into a wider establishing shot. Reveal the entire futuristic silver starship in the snowy alien plain, including the nose, full hull, both engines, landing legs, surrounding snow, and icy cliffs. Keep the same starship identity and material. Do not crop the ship. Keep it sharp and centered in a coherent wide frame." \
  --negative "close-up, cropped ship, cut off nose, cut off engines, duplicated spaceship, blurry ship, text, watermark" \
  --steps 20 \
  --guidance 4 \
  --seed 8301 \
  --metadata \
  --replace \
  --output validation_outputs/reframe_outpaint_matrix_2026_06_08/qwen2509_q4_reframe_b.png
```

### Outpaint B

```sh
uv run mlxgen generate \
  --model AbstractFramework/qwen-image-edit-2509-4bit \
  --image docs/assets/validation/reframe-outpaint-2026-06-08/source-b-cropped-starship.png \
  --outpaint-padding "5%,80%,5%,60%" \
  --prompt "Outpaint this close cropped starship image into a much wider realistic shot of the full spacecraft in the snowy canyon. Keep the existing central spacecraft surface consistent, and complete the missing nose, full hull, tail, engines, snow field, and ice cliffs in the newly added space. The entire ship must fit inside the final wide frame with empty snow visible around it. Preserve the same lighting and camera angle. No text, no frame, no border, no duplicate ship." \
  --negative "text, border, frame, hard seam, split image, collage, duplicate spacecraft, duplicated mountains, repeated mountain peaks, distorted engines, melted hull, blurry ship, cropped ship, cut off hull, cut off engines" \
  --steps 20 \
  --guidance 4 \
  --seed 8312 \
  --metadata \
  --replace \
  --output validation_outputs/reframe_outpaint_matrix_2026_06_08/qwen2509_q4_outpaint_b.png
```

## Qwen Image Edit 2511 / source

### Reframe B

```sh
uv run mlxgen generate \
  --model Qwen/Qwen-Image-Edit-2511 \
  --image docs/assets/validation/reframe-outpaint-2026-06-08/source-b-cropped-starship.png \
  --reframe-padding "25%,50%,25%,50%" \
  --prompt "Generatively reframe this close-up into a wider establishing shot. Reveal the entire futuristic silver starship in the snowy alien plain, including the nose, full hull, both engines, landing legs, surrounding snow, and icy cliffs. Keep the same starship identity and material. Do not crop the ship. Keep it sharp and centered in a coherent wide frame." \
  --negative "close-up, cropped ship, cut off nose, cut off engines, duplicated spaceship, blurry ship, text, watermark" \
  --steps 20 \
  --guidance 4 \
  --seed 8401 \
  --metadata \
  --replace \
  --output validation_outputs/reframe_outpaint_matrix_2026_06_08/qwen2511_source_reframe_b.png
```

### Outpaint B

```sh
uv run mlxgen generate \
  --model Qwen/Qwen-Image-Edit-2511 \
  --image docs/assets/validation/reframe-outpaint-2026-06-08/source-b-cropped-starship.png \
  --outpaint-padding "5%,80%,5%,60%" \
  --prompt "Outpaint this close cropped image into a wider realistic snowy canyon shot while keeping the same compact pod-like silver starship design from the source. Complete the missing nose, rounded hull, short tail, twin round rear engines, snow field, and ice cliffs in the newly added space. The final ship must remain a compact rounded spacecraft, not an airplane, with no large wings. Preserve the same lighting and camera angle. No text, no frame, no border, no duplicate ship." \
  --negative "airplane, jet aircraft, long wing, black wing, flat wing, runway, text, border, frame, hard seam, split image, collage, duplicate spacecraft, duplicated mountains, repeated mountain peaks, distorted engines, melted hull, blurry ship, cropped ship, cut off hull, cut off engines" \
  --steps 20 \
  --guidance 4 \
  --seed 8413 \
  --metadata \
  --replace \
  --output validation_outputs/reframe_outpaint_matrix_2026_06_08/qwen2511_source_outpaint_b_retry_compact.png
```

## Qwen Image Edit 2511 / q8

### Reframe B

```sh
uv run mlxgen generate \
  --model AbstractFramework/qwen-image-edit-2511-8bit \
  --image docs/assets/validation/reframe-outpaint-2026-06-08/source-b-cropped-starship.png \
  --reframe-padding "25%,50%,25%,50%" \
  --prompt "Generatively reframe this close-up into a wider establishing shot. Reveal the entire futuristic silver starship in the snowy alien plain, including the nose, full hull, both engines, landing legs, surrounding snow, and icy cliffs. Keep the same starship identity and material. Do not crop the ship. Keep it sharp and centered in a coherent wide frame." \
  --negative "close-up, cropped ship, cut off nose, cut off engines, duplicated spaceship, blurry ship, text, watermark" \
  --steps 20 \
  --guidance 4 \
  --seed 8401 \
  --metadata \
  --replace \
  --output validation_outputs/reframe_outpaint_matrix_2026_06_08/qwen2511_q8_reframe_b.png
```

### Outpaint B

```sh
uv run mlxgen generate \
  --model AbstractFramework/qwen-image-edit-2511-8bit \
  --image docs/assets/validation/reframe-outpaint-2026-06-08/source-b-cropped-starship.png \
  --outpaint-padding "5%,80%,5%,60%" \
  --prompt "Outpaint this close cropped image into a wider realistic snowy canyon shot while keeping the same compact pod-like silver starship design from the source. Complete the missing nose, rounded hull, short tail, twin round rear engines, snow field, and ice cliffs in the newly added space. The final ship must remain a compact rounded spacecraft, not an airplane, with no large wings. Preserve the same lighting and camera angle. No text, no frame, no border, no duplicate ship." \
  --negative "airplane, jet aircraft, long wing, black wing, flat wing, runway, text, border, frame, hard seam, split image, collage, duplicate spacecraft, duplicated mountains, repeated mountain peaks, distorted engines, melted hull, blurry ship, cropped ship, cut off hull, cut off engines" \
  --steps 20 \
  --guidance 4 \
  --seed 8413 \
  --metadata \
  --replace \
  --output validation_outputs/reframe_outpaint_matrix_2026_06_08/qwen2511_q8_outpaint_b.png
```

## Qwen Image Edit 2511 / q4

### Reframe B

```sh
uv run mlxgen generate \
  --model AbstractFramework/qwen-image-edit-2511-4bit \
  --image docs/assets/validation/reframe-outpaint-2026-06-08/source-b-cropped-starship.png \
  --reframe-padding "25%,50%,25%,50%" \
  --prompt "Generatively reframe this close-up into a wider establishing shot. Reveal the entire futuristic silver starship in the snowy alien plain, including the nose, full hull, both engines, landing legs, surrounding snow, and icy cliffs. Keep the same starship identity and material. Do not crop the ship. Keep it sharp and centered in a coherent wide frame." \
  --negative "close-up, cropped ship, cut off nose, cut off engines, duplicated spaceship, blurry ship, text, watermark" \
  --steps 20 \
  --guidance 4 \
  --seed 8401 \
  --metadata \
  --replace \
  --output validation_outputs/reframe_outpaint_matrix_2026_06_08/qwen2511_q4_reframe_b.png
```

### Outpaint B

```sh
uv run mlxgen generate \
  --model AbstractFramework/qwen-image-edit-2511-4bit \
  --image docs/assets/validation/reframe-outpaint-2026-06-08/source-b-cropped-starship.png \
  --outpaint-padding "5%,80%,5%,60%" \
  --prompt "Outpaint this close cropped image into a wider realistic snowy canyon shot while keeping the same compact pod-like silver starship design from the source. Complete the missing nose, rounded hull, short tail, twin round rear engines, snow field, and ice cliffs in the newly added space. The final ship must remain a compact rounded spacecraft, not an airplane, with no large wings. Preserve the same lighting and camera angle. No text, no frame, no border, no duplicate ship." \
  --negative "airplane, jet aircraft, long wing, black wing, flat wing, runway, text, border, frame, hard seam, split image, collage, duplicate spacecraft, duplicated mountains, repeated mountain peaks, distorted engines, melted hull, blurry ship, cropped ship, cut off hull, cut off engines" \
  --steps 20 \
  --guidance 4 \
  --seed 8413 \
  --metadata \
  --replace \
  --output validation_outputs/reframe_outpaint_matrix_2026_06_08/qwen2511_q4_outpaint_b.png
```

## FLUX.2 Klein 4B / source

### Reframe B

```sh
uv run mlxgen generate \
  --model black-forest-labs/FLUX.2-klein-4B \
  --image docs/assets/validation/reframe-outpaint-2026-06-08/source-b-cropped-starship.png \
  --reframe-padding "25%,50%,25%,50%" \
  --prompt "Generatively reframe this close-up into a wider establishing shot. Reveal the entire futuristic silver starship in the snowy alien plain, including the nose, full hull, both engines, landing legs, surrounding snow, and icy cliffs. Keep the same starship identity and material. Do not crop the ship. Keep it sharp and centered in a coherent wide frame. No duplicated spacecraft, no text, no border." \
  --steps 16 \
  --guidance 1 \
  --seed 8501 \
  --metadata \
  --replace \
  --output validation_outputs/reframe_outpaint_matrix_2026_06_08/flux2_4b_source_reframe_b.png
```

### Outpaint B

```sh
uv run mlxgen generate \
  --model black-forest-labs/FLUX.2-klein-4B \
  --image docs/assets/validation/reframe-outpaint-2026-06-08/source-b-cropped-starship.png \
  --outpaint-padding "5%,80%,5%,60%" \
  --prompt "Outpaint this close cropped starship image into a much wider realistic shot of the full spacecraft in the snowy canyon. Keep the existing compact silver spacecraft consistent, complete the missing nose, rounded hull, short tail, twin round rear engines, snow field, and ice cliffs in the newly added space. The entire ship must fit inside the final wide frame. No duplicated spacecraft, no repeated mountains, no text, no border." \
  --steps 16 \
  --guidance 1 \
  --seed 8512 \
  --metadata \
  --replace \
  --output validation_outputs/reframe_outpaint_matrix_2026_06_08/flux2_4b_source_outpaint_b.png
```

## FLUX.2 Klein 4B / q8

### Reframe B

```sh
uv run mlxgen generate \
  --model AbstractFramework/flux.2-klein-4b-8bit \
  --image docs/assets/validation/reframe-outpaint-2026-06-08/source-b-cropped-starship.png \
  --reframe-padding "25%,50%,25%,50%" \
  --prompt "Generatively reframe this close-up into a wider establishing shot. Reveal the entire futuristic silver starship in the snowy alien plain, including the nose, full hull, both engines, landing legs, surrounding snow, and icy cliffs. Keep the same starship identity and material. Do not crop the ship. Keep it sharp and centered in a coherent wide frame. No duplicated spacecraft, no text, no border." \
  --steps 16 \
  --guidance 1 \
  --seed 8501 \
  --metadata \
  --replace \
  --output validation_outputs/reframe_outpaint_matrix_2026_06_08/flux2_4b_q8_reframe_b.png
```

### Outpaint B

```sh
uv run mlxgen generate \
  --model AbstractFramework/flux.2-klein-4b-8bit \
  --image docs/assets/validation/reframe-outpaint-2026-06-08/source-b-cropped-starship.png \
  --outpaint-padding "5%,80%,5%,60%" \
  --prompt "Outpaint this close cropped starship image into a much wider realistic shot of the full spacecraft in the snowy canyon. Keep the existing compact silver spacecraft consistent, complete the missing nose, rounded hull, short tail, twin round rear engines, snow field, and ice cliffs in the newly added space. The entire ship must fit inside the final wide frame. No duplicated spacecraft, no repeated mountains, no text, no border." \
  --steps 16 \
  --guidance 1 \
  --seed 8512 \
  --metadata \
  --replace \
  --output validation_outputs/reframe_outpaint_matrix_2026_06_08/flux2_4b_q8_outpaint_b.png
```

## FLUX.2 Klein 4B / q4

### Reframe B

```sh
uv run mlxgen generate \
  --model AbstractFramework/flux.2-klein-4b-4bit \
  --image docs/assets/validation/reframe-outpaint-2026-06-08/source-b-cropped-starship.png \
  --reframe-padding "25%,50%,25%,50%" \
  --prompt "Generatively reframe this close-up into a wider establishing shot. Reveal the entire futuristic silver starship in the snowy alien plain, including the nose, full hull, both engines, landing legs, surrounding snow, and icy cliffs. Keep the same starship identity and material. Do not crop the ship. Keep it sharp and centered in a coherent wide frame. No duplicated spacecraft, no text, no border." \
  --steps 16 \
  --guidance 1 \
  --seed 8501 \
  --metadata \
  --replace \
  --output validation_outputs/reframe_outpaint_matrix_2026_06_08/flux2_4b_q4_reframe_b.png
```

### Outpaint B

```sh
uv run mlxgen generate \
  --model AbstractFramework/flux.2-klein-4b-4bit \
  --image docs/assets/validation/reframe-outpaint-2026-06-08/source-b-cropped-starship.png \
  --outpaint-padding "5%,80%,5%,60%" \
  --prompt "Outpaint this close cropped starship image into a much wider realistic shot of the full spacecraft in the snowy canyon. Keep the existing compact silver spacecraft consistent, complete the missing nose, rounded hull, short tail, twin round rear engines, snow field, and ice cliffs in the newly added space. The entire ship must fit inside the final wide frame. No duplicated spacecraft, no repeated mountains, no text, no border." \
  --steps 16 \
  --guidance 1 \
  --seed 8512 \
  --metadata \
  --replace \
  --output validation_outputs/reframe_outpaint_matrix_2026_06_08/flux2_4b_q4_outpaint_b.png
```

## FLUX.2 Klein 9B / source

### Reframe B

```sh
uv run mlxgen generate \
  --model black-forest-labs/FLUX.2-klein-9B \
  --image docs/assets/validation/reframe-outpaint-2026-06-08/source-b-cropped-starship.png \
  --reframe-padding "25%,80%,25%,60%" \
  --prompt "Zoom out from the source image into a wider snowy canyon view while keeping the exact same visible spacecraft design: a smooth silver sci-fi hull seen from the side, pointed nose on the left, one large circular black side engine intake, rounded metal body, short landing legs, and snowy canyon background. Use the larger canvas to reveal the missing rear, tail, full hull, surrounding snow, and ice cliffs. Keep the original side-view camera angle. Do not redesign it as an airplane, do not add long wings, propellers, or a front-facing cockpit aircraft view. No duplicate ship, no text, no border." \
  --steps 16 \
  --guidance 1 \
  --seed 8604 \
  --metadata \
  --replace \
  --output validation_outputs/reframe_outpaint_matrix_2026_06_08/flux2_9b_source_reframe_b_wide_anchors.png
```

### Outpaint B

```sh
uv run mlxgen generate \
  --model black-forest-labs/FLUX.2-klein-9B \
  --image docs/assets/validation/reframe-outpaint-2026-06-08/source-b-cropped-starship.png \
  --outpaint-padding "5%,80%,5%,60%" \
  --prompt "Outpaint this close cropped starship image into a much wider realistic shot of the full spacecraft in the snowy canyon. Keep the existing compact silver spacecraft consistent, complete the missing nose, rounded hull, short tail, twin round rear engines, snow field, and ice cliffs in the newly added space. The entire ship must fit inside the final wide frame. No duplicated spacecraft, no repeated mountains, no text, no border." \
  --steps 16 \
  --guidance 1 \
  --seed 8612 \
  --metadata \
  --replace \
  --output validation_outputs/reframe_outpaint_matrix_2026_06_08/flux2_9b_source_outpaint_b.png
```

## FLUX.2 Klein 9B / q8

### Reframe B

```sh
uv run mlxgen generate \
  --model AbstractFramework/flux.2-klein-9b-8bit \
  --image docs/assets/validation/reframe-outpaint-2026-06-08/source-b-cropped-starship.png \
  --reframe-padding "25%,80%,25%,60%" \
  --prompt "Zoom out from the source image into a wider snowy canyon view while keeping the exact same visible spacecraft design: a smooth silver sci-fi hull seen from the side, pointed nose on the left, one large circular black side engine intake, rounded metal body, short landing legs, and snowy canyon background. Use the larger canvas to reveal the missing rear, tail, full hull, surrounding snow, and ice cliffs. Keep the original side-view camera angle. Do not redesign it as an airplane, do not add long wings, propellers, or a front-facing cockpit aircraft view. No duplicate ship, no text, no border." \
  --steps 16 \
  --guidance 1 \
  --seed 8604 \
  --metadata \
  --replace \
  --output validation_outputs/reframe_outpaint_matrix_2026_06_08/flux2_9b_q8_reframe_b.png
```

### Outpaint B

```sh
uv run mlxgen generate \
  --model AbstractFramework/flux.2-klein-9b-8bit \
  --image docs/assets/validation/reframe-outpaint-2026-06-08/source-b-cropped-starship.png \
  --outpaint-padding "5%,80%,5%,60%" \
  --prompt "Outpaint this close cropped starship image into a much wider realistic shot of the full spacecraft in the snowy canyon. Keep the existing compact silver spacecraft consistent, complete the missing nose, rounded hull, short tail, twin round rear engines, snow field, and ice cliffs in the newly added space. The entire ship must fit inside the final wide frame. No duplicated spacecraft, no repeated mountains, no text, no border." \
  --steps 16 \
  --guidance 1 \
  --seed 8612 \
  --metadata \
  --replace \
  --output validation_outputs/reframe_outpaint_matrix_2026_06_08/flux2_9b_q8_outpaint_b.png
```

## FLUX.2 Klein 9B / q4

### Reframe B

```sh
uv run mlxgen generate \
  --model AbstractFramework/flux.2-klein-9b-4bit \
  --image docs/assets/validation/reframe-outpaint-2026-06-08/source-b-cropped-starship.png \
  --reframe-padding "25%,80%,25%,60%" \
  --prompt "Zoom out from the source image into a wider snowy canyon view while keeping the exact same visible spacecraft design: a smooth silver sci-fi hull seen from the side, pointed nose on the left, one large circular black side engine intake, rounded metal body, short landing legs, and snowy canyon background. Use the larger canvas to reveal the missing rear, tail, full hull, surrounding snow, and ice cliffs. Keep the original side-view camera angle. Do not redesign it as an airplane, do not add long wings, propellers, or a front-facing cockpit aircraft view. No duplicate ship, no text, no border." \
  --steps 16 \
  --guidance 1 \
  --seed 8604 \
  --metadata \
  --replace \
  --output validation_outputs/reframe_outpaint_matrix_2026_06_08/flux2_9b_q4_reframe_b.png
```

### Outpaint B

```sh
uv run mlxgen generate \
  --model AbstractFramework/flux.2-klein-9b-4bit \
  --image docs/assets/validation/reframe-outpaint-2026-06-08/source-b-cropped-starship.png \
  --outpaint-padding "5%,80%,5%,60%" \
  --prompt "Outpaint this close cropped starship image into a much wider realistic shot of the full spacecraft in the snowy canyon. Keep the existing compact silver spacecraft consistent, complete the missing nose, rounded hull, short tail, twin round rear engines, snow field, and ice cliffs in the newly added space. The entire ship must fit inside the final wide frame. No duplicated spacecraft, no repeated mountains, no text, no border." \
  --steps 16 \
  --guidance 1 \
  --seed 8612 \
  --metadata \
  --replace \
  --output validation_outputs/reframe_outpaint_matrix_2026_06_08/flux2_9b_q4_outpaint_b.png
```
