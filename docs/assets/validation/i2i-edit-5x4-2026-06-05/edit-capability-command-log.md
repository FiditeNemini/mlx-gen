# Edit Capability Validation Commands

These are the exact `mlxgen generate` commands used for the published 5x4 edit matrix rows for validated FLUX.2 Klein and Qwen Image Edit 2509 models. The matrix source image is `docs/assets/examples/spaceship-snow/01_t2i_spaceship_snow.png`.

Column meanings:

- `B`: cinematic latent/style variation.
- `C`: crash edit from the source image.
- `D`: pencil sketch edit.
- `E`: multi-reference composition from the model's own pencil/crash and cinematic rows.

### FLUX.2 Klein 4B / source / B

```sh
uv run mlxgen generate --model black-forest-labs/FLUX.2-klein-4B --prompt "Make this same spaceship in the snow look like polished cinematic science-fiction concept art at blue hour. Preserve the exact camera angle, ship position, snowy canyon, and overall layout. Sharpen hull panels and add cold blue shadows; no crash, no damage." --width 432 --height 240 --steps 20 --seed 9401 --metadata --replace --output validation_outputs/edit_prepared_capability_2026_06_05/flux2_klein_4b_source_b_cinematic.png --image docs/assets/examples/spaceship-snow/01_t2i_spaceship_snow.png --i2i-mode latent --image-strength 0.35
```

### FLUX.2 Klein 4B / source / C

```sh
uv run mlxgen generate --model black-forest-labs/FLUX.2-klein-4B --prompt "Edit the source into the same spaceship after a hard landing in the snow. Preserve the same camera angle, spaceship identity, rear engines, canyon cliffs, and wide framing. The ship must remain solid and sharp, but show a tilted hull, bent landing struts, broken ice chunks, disturbed snow, a shallow scrape trail, and a thin smoke plume. No blur, no mesh, no dissolve." --width 432 --height 240 --steps 20 --seed 9402 --metadata --replace --output validation_outputs/edit_prepared_capability_2026_06_05/flux2_klein_4b_source_c_crash.png --image docs/assets/examples/spaceship-snow/01_t2i_spaceship_snow.png --i2i-mode edit
```

### FLUX.2 Klein 4B / source / D

```sh
uv run mlxgen generate --model black-forest-labs/FLUX.2-klein-4B --prompt "Turn the source into a clean graphite pencil sketch of the same hard-landed spaceship scene. Preserve the spaceship identity, snowy canyon layout, tilted hull, bent landing struts, disturbed snow, debris, and smoke plume. White paper background, precise line art, no color fill, no blur." --width 432 --height 240 --steps 20 --seed 9403 --metadata --replace --output validation_outputs/edit_prepared_capability_2026_06_05/flux2_klein_4b_source_d_pencil_crash.png --image docs/assets/examples/spaceship-snow/01_t2i_spaceship_snow.png --i2i-mode edit
```

### FLUX.2 Klein 4B / source / E

```sh
uv run mlxgen generate --model black-forest-labs/FLUX.2-klein-4B --prompt "Use the first image as the pencil crash structure and the second image as the cinematic lighting and color reference. Produce one coherent image of the same hard-landed spaceship in the snow: graphite sketch lines with subtle cinematic blue-hour shading, stable canyon layout, solid spaceship, visible crash debris, no blur, no text." --width 432 --height 240 --steps 20 --seed 9404 --metadata --replace --output validation_outputs/edit_prepared_capability_2026_06_05/flux2_klein_4b_source_e_composition.png --image validation_outputs/edit_prepared_capability_2026_06_05/flux2_klein_4b_source_d_pencil_crash.png --image validation_outputs/edit_prepared_capability_2026_06_05/flux2_klein_4b_source_b_cinematic.png --i2i-mode multi-reference
```

### FLUX.2 Klein 4B / q8 prepared / B

```sh
uv run mlxgen generate --model AbstractFramework/flux.2-klein-4b-8bit --prompt "Make this same spaceship in the snow look like polished cinematic science-fiction concept art at blue hour. Preserve the exact camera angle, ship position, snowy canyon, and overall layout. Sharpen hull panels and add cold blue shadows; no crash, no damage." --width 432 --height 240 --steps 20 --seed 9201 --metadata --replace --output validation_outputs/i2i_standard_sequence_2026_06_05/flux2_klein_4b_8bit_b_cinematic.png --image docs/assets/examples/spaceship-snow/01_t2i_spaceship_snow.png --i2i-mode latent --image-strength 0.35
```

### FLUX.2 Klein 4B / q8 prepared / C

```sh
uv run mlxgen generate --model AbstractFramework/flux.2-klein-4b-8bit --prompt "Edit the source into the same spaceship after a hard landing in the snow. Preserve the same camera angle, spaceship identity, rear engines, canyon cliffs, and wide framing. The ship must remain solid and sharp, but show a tilted hull, bent landing struts, broken ice chunks, disturbed snow, a shallow scrape trail, and a thin smoke plume. No blur, no mesh, no dissolve." --width 432 --height 240 --steps 20 --seed 9202 --metadata --replace --output validation_outputs/i2i_standard_sequence_2026_06_05/flux2_klein_4b_8bit_c_crash.png --image docs/assets/examples/spaceship-snow/01_t2i_spaceship_snow.png --i2i-mode edit
```

### FLUX.2 Klein 4B / q8 prepared / D

```sh
uv run mlxgen generate --model AbstractFramework/flux.2-klein-4b-8bit --prompt "Turn the source into a clean graphite pencil sketch of the same hard-landed spaceship scene. Preserve the spaceship identity, snowy canyon layout, tilted hull, bent landing struts, disturbed snow, debris, and smoke plume. White paper background, precise line art, no color fill, no blur." --width 432 --height 240 --steps 20 --seed 9203 --metadata --replace --output validation_outputs/i2i_standard_sequence_2026_06_05/flux2_klein_4b_8bit_d_pencil_crash.png --image docs/assets/examples/spaceship-snow/01_t2i_spaceship_snow.png --i2i-mode edit
```

### FLUX.2 Klein 4B / q8 prepared / E

```sh
uv run mlxgen generate --model AbstractFramework/flux.2-klein-4b-8bit --prompt "Use the first image as the pencil crash structure and the second image as the cinematic lighting and color reference. Produce one coherent image of the same hard-landed spaceship in the snow: graphite sketch lines with subtle cinematic blue-hour shading, stable canyon layout, solid spaceship, visible crash debris, no blur, no text." --width 432 --height 240 --steps 20 --seed 9204 --metadata --replace --output validation_outputs/i2i_standard_sequence_2026_06_05/flux2_klein_4b_8bit_e_composition.png --image validation_outputs/i2i_standard_sequence_2026_06_05/flux2_klein_4b_8bit_d_pencil_crash.png --image validation_outputs/i2i_standard_sequence_2026_06_05/flux2_klein_4b_8bit_b_cinematic.png --i2i-mode multi-reference
```

### FLUX.2 Klein 4B / q4 prepared / B

```sh
uv run mlxgen generate --model AbstractFramework/flux.2-klein-4b-4bit --prompt "Make this same spaceship in the snow look like polished cinematic science-fiction concept art at blue hour. Preserve the exact camera angle, ship position, snowy canyon, and overall layout. Sharpen hull panels and add cold blue shadows; no crash, no damage." --width 432 --height 240 --steps 20 --seed 9201 --metadata --replace --output validation_outputs/i2i_standard_sequence_2026_06_05/flux2_klein_4b_4bit_b_cinematic.png --image docs/assets/examples/spaceship-snow/01_t2i_spaceship_snow.png --i2i-mode latent --image-strength 0.35
```

### FLUX.2 Klein 4B / q4 prepared / C

```sh
uv run mlxgen generate --model AbstractFramework/flux.2-klein-4b-4bit --prompt "Edit the source into the same spaceship after a hard landing in the snow. Preserve the same camera angle, spaceship identity, rear engines, canyon cliffs, and wide framing. The ship must remain solid and sharp, but show a tilted hull, bent landing struts, broken ice chunks, disturbed snow, a shallow scrape trail, and a thin smoke plume. No blur, no mesh, no dissolve." --width 432 --height 240 --steps 20 --seed 9202 --metadata --replace --output validation_outputs/i2i_standard_sequence_2026_06_05/flux2_klein_4b_4bit_c_crash.png --image docs/assets/examples/spaceship-snow/01_t2i_spaceship_snow.png --i2i-mode edit
```

### FLUX.2 Klein 4B / q4 prepared / D

```sh
uv run mlxgen generate --model AbstractFramework/flux.2-klein-4b-4bit --prompt "Turn the source into a clean graphite pencil sketch of the same hard-landed spaceship scene. Preserve the spaceship identity, snowy canyon layout, tilted hull, bent landing struts, disturbed snow, debris, and smoke plume. White paper background, precise line art, no color fill, no blur." --width 432 --height 240 --steps 20 --seed 9203 --metadata --replace --output validation_outputs/i2i_standard_sequence_2026_06_05/flux2_klein_4b_4bit_d_pencil_crash.png --image docs/assets/examples/spaceship-snow/01_t2i_spaceship_snow.png --i2i-mode edit
```

### FLUX.2 Klein 4B / q4 prepared / E

```sh
uv run mlxgen generate --model AbstractFramework/flux.2-klein-4b-4bit --prompt "Use the first image as the pencil crash structure and the second image as the cinematic lighting and color reference. Produce one coherent image of the same hard-landed spaceship in the snow: graphite sketch lines with subtle cinematic blue-hour shading, stable canyon layout, solid spaceship, visible crash debris, no blur, no text." --width 432 --height 240 --steps 20 --seed 9204 --metadata --replace --output validation_outputs/i2i_standard_sequence_2026_06_05/flux2_klein_4b_4bit_e_composition.png --image validation_outputs/i2i_standard_sequence_2026_06_05/flux2_klein_4b_4bit_d_pencil_crash.png --image validation_outputs/i2i_standard_sequence_2026_06_05/flux2_klein_4b_4bit_b_cinematic.png --i2i-mode multi-reference
```

### FLUX.2 Klein 9B / source / B

```sh
uv run mlxgen generate --model black-forest-labs/FLUX.2-klein-9B --prompt "Make this same spaceship in the snow look like polished cinematic science-fiction concept art at blue hour. Preserve the exact camera angle, ship position, snowy canyon, and overall layout. Sharpen hull panels and add cold blue shadows; no crash, no damage." --width 432 --height 240 --steps 20 --seed 9411 --metadata --replace --output validation_outputs/edit_prepared_capability_2026_06_05/flux2_klein_9b_source_b_cinematic.png --image docs/assets/examples/spaceship-snow/01_t2i_spaceship_snow.png --i2i-mode latent --image-strength 0.35
```

### FLUX.2 Klein 9B / source / C

```sh
uv run mlxgen generate --model black-forest-labs/FLUX.2-klein-9B --prompt "Edit the source into the same spaceship after a hard landing in the snow. Preserve the same camera angle, spaceship identity, rear engines, canyon cliffs, and wide framing. The ship must remain solid and sharp, but show a tilted hull, bent landing struts, broken ice chunks, disturbed snow, a shallow scrape trail, and a thin smoke plume. No blur, no mesh, no dissolve." --width 432 --height 240 --steps 20 --seed 9412 --metadata --replace --output validation_outputs/edit_prepared_capability_2026_06_05/flux2_klein_9b_source_c_crash.png --image docs/assets/examples/spaceship-snow/01_t2i_spaceship_snow.png --i2i-mode edit
```

### FLUX.2 Klein 9B / source / D

```sh
uv run mlxgen generate --model black-forest-labs/FLUX.2-klein-9B --prompt "Turn the source into a clean graphite pencil sketch of the same hard-landed spaceship scene. Preserve the spaceship identity, snowy canyon layout, tilted hull, bent landing struts, disturbed snow, debris, and smoke plume. White paper background, precise line art, no color fill, no blur." --width 432 --height 240 --steps 20 --seed 9413 --metadata --replace --output validation_outputs/edit_prepared_capability_2026_06_05/flux2_klein_9b_source_d_pencil_crash.png --image docs/assets/examples/spaceship-snow/01_t2i_spaceship_snow.png --i2i-mode edit
```

### FLUX.2 Klein 9B / source / E

```sh
uv run mlxgen generate --model black-forest-labs/FLUX.2-klein-9B --prompt "Use the first image as the pencil crash structure and the second image as the cinematic lighting and color reference. Produce one coherent image of the same hard-landed spaceship in the snow: graphite sketch lines with subtle cinematic blue-hour shading, stable canyon layout, solid spaceship, visible crash debris, no blur, no text." --width 432 --height 240 --steps 20 --seed 9414 --metadata --replace --output validation_outputs/edit_prepared_capability_2026_06_05/flux2_klein_9b_source_e_composition.png --image validation_outputs/edit_prepared_capability_2026_06_05/flux2_klein_9b_source_d_pencil_crash.png --image validation_outputs/edit_prepared_capability_2026_06_05/flux2_klein_9b_source_b_cinematic.png --i2i-mode multi-reference
```

### FLUX.2 Klein 9B / q8 prepared / B

```sh
uv run mlxgen generate --model AbstractFramework/flux.2-klein-9b-8bit --prompt "Make this same spaceship in the snow look like polished cinematic science-fiction concept art at blue hour. Preserve the exact camera angle, ship position, snowy canyon, and overall layout. Sharpen hull panels and add cold blue shadows; no crash, no damage." --width 432 --height 240 --steps 20 --seed 9201 --metadata --replace --output validation_outputs/i2i_standard_sequence_2026_06_05/flux2_klein_9b_8bit_b_cinematic.png --image docs/assets/examples/spaceship-snow/01_t2i_spaceship_snow.png --i2i-mode latent --image-strength 0.35
```

### FLUX.2 Klein 9B / q8 prepared / C

```sh
uv run mlxgen generate --model AbstractFramework/flux.2-klein-9b-8bit --prompt "Edit the source into the same spaceship after a hard landing in the snow. Preserve the same camera angle, spaceship identity, rear engines, canyon cliffs, and wide framing. The ship must remain solid and sharp, but show a tilted hull, bent landing struts, broken ice chunks, disturbed snow, a shallow scrape trail, and a thin smoke plume. No blur, no mesh, no dissolve." --width 432 --height 240 --steps 20 --seed 9202 --metadata --replace --output validation_outputs/i2i_standard_sequence_2026_06_05/flux2_klein_9b_8bit_c_crash.png --image docs/assets/examples/spaceship-snow/01_t2i_spaceship_snow.png --i2i-mode edit
```

### FLUX.2 Klein 9B / q8 prepared / D

```sh
uv run mlxgen generate --model AbstractFramework/flux.2-klein-9b-8bit --prompt "Turn the source into a clean graphite pencil sketch of the same hard-landed spaceship scene. Preserve the spaceship identity, snowy canyon layout, tilted hull, bent landing struts, disturbed snow, debris, and smoke plume. White paper background, precise line art, no color fill, no blur." --width 432 --height 240 --steps 20 --seed 9203 --metadata --replace --output validation_outputs/i2i_standard_sequence_2026_06_05/flux2_klein_9b_8bit_d_pencil_crash.png --image docs/assets/examples/spaceship-snow/01_t2i_spaceship_snow.png --i2i-mode edit
```

### FLUX.2 Klein 9B / q8 prepared / E

```sh
uv run mlxgen generate --model AbstractFramework/flux.2-klein-9b-8bit --prompt "Use the first image as the pencil crash structure and the second image as the cinematic lighting and color reference. Produce one coherent image of the same hard-landed spaceship in the snow: graphite sketch lines with subtle cinematic blue-hour shading, stable canyon layout, solid spaceship, visible crash debris, no blur, no text." --width 432 --height 240 --steps 20 --seed 9204 --metadata --replace --output validation_outputs/i2i_standard_sequence_2026_06_05/flux2_klein_9b_8bit_e_composition.png --image validation_outputs/i2i_standard_sequence_2026_06_05/flux2_klein_9b_8bit_d_pencil_crash.png --image validation_outputs/i2i_standard_sequence_2026_06_05/flux2_klein_9b_8bit_b_cinematic.png --i2i-mode multi-reference
```

### FLUX.2 Klein 9B / q4 prepared / B

```sh
uv run mlxgen generate --model AbstractFramework/flux.2-klein-9b-4bit --prompt "Make this same spaceship in the snow look like polished cinematic science-fiction concept art at blue hour. Preserve the exact camera angle, ship position, snowy canyon, and overall layout. Sharpen hull panels and add cold blue shadows; no crash, no damage." --width 432 --height 240 --steps 20 --seed 9201 --metadata --replace --output validation_outputs/i2i_standard_sequence_2026_06_05/flux2_klein_9b_4bit_b_cinematic.png --image docs/assets/examples/spaceship-snow/01_t2i_spaceship_snow.png --i2i-mode latent --image-strength 0.35
```

### FLUX.2 Klein 9B / q4 prepared / C

```sh
uv run mlxgen generate --model AbstractFramework/flux.2-klein-9b-4bit --prompt "Edit the source into the same spaceship after a hard landing in the snow. Preserve the same camera angle, spaceship identity, rear engines, canyon cliffs, and wide framing. The ship must remain solid and sharp, but show a tilted hull, bent landing struts, broken ice chunks, disturbed snow, a shallow scrape trail, and a thin smoke plume. No blur, no mesh, no dissolve." --width 432 --height 240 --steps 20 --seed 9202 --metadata --replace --output validation_outputs/i2i_standard_sequence_2026_06_05/flux2_klein_9b_4bit_c_crash.png --image docs/assets/examples/spaceship-snow/01_t2i_spaceship_snow.png --i2i-mode edit
```

### FLUX.2 Klein 9B / q4 prepared / D

```sh
uv run mlxgen generate --model AbstractFramework/flux.2-klein-9b-4bit --prompt "Turn the source into a clean graphite pencil sketch of the same hard-landed spaceship scene. Preserve the spaceship identity, snowy canyon layout, tilted hull, bent landing struts, disturbed snow, debris, and smoke plume. White paper background, precise line art, no color fill, no blur." --width 432 --height 240 --steps 20 --seed 9203 --metadata --replace --output validation_outputs/i2i_standard_sequence_2026_06_05/flux2_klein_9b_4bit_d_pencil_crash.png --image docs/assets/examples/spaceship-snow/01_t2i_spaceship_snow.png --i2i-mode edit
```

### FLUX.2 Klein 9B / q4 prepared / E

```sh
uv run mlxgen generate --model AbstractFramework/flux.2-klein-9b-4bit --prompt "Use the first image as the pencil crash structure and the second image as the cinematic lighting and color reference. Produce one coherent image of the same hard-landed spaceship in the snow: graphite sketch lines with subtle cinematic blue-hour shading, stable canyon layout, solid spaceship, visible crash debris, no blur, no text." --width 432 --height 240 --steps 20 --seed 9204 --metadata --replace --output validation_outputs/i2i_standard_sequence_2026_06_05/flux2_klein_9b_4bit_e_composition.png --image validation_outputs/i2i_standard_sequence_2026_06_05/flux2_klein_9b_4bit_d_pencil_crash.png --image validation_outputs/i2i_standard_sequence_2026_06_05/flux2_klein_9b_4bit_b_cinematic.png --i2i-mode multi-reference
```

### Qwen Image Edit 2509 / source / B

```sh
uv run mlxgen generate --model Qwen/Qwen-Image-Edit-2509 --prompt "Make this same spaceship in the snow look like polished cinematic science-fiction concept art at blue hour. Preserve the exact camera angle, ship position, snowy canyon, and overall layout. Sharpen hull panels and add cold blue shadows; no crash, no damage." --width 432 --height 240 --steps 20 --seed 9201 --metadata --replace --output validation_outputs/edit_prepared_capability_2026_06_05/qwen_edit_2509_source_b_cinematic.png --image docs/assets/examples/spaceship-snow/01_t2i_spaceship_snow.png --i2i-mode edit
```

### Qwen Image Edit 2509 / source / C

```sh
uv run mlxgen generate --model Qwen/Qwen-Image-Edit-2509 --prompt "Edit the source into the same spaceship after a hard landing in the snow. Preserve the same camera angle, spaceship identity, rear engines, canyon cliffs, and wide framing. The ship must remain solid and sharp, but show a tilted hull, bent landing struts, broken ice chunks, disturbed snow, a shallow scrape trail, and a thin smoke plume. No blur, no mesh, no dissolve." --width 432 --height 240 --steps 20 --seed 9202 --metadata --replace --output validation_outputs/edit_prepared_capability_2026_06_05/qwen_edit_2509_source_c_crash.png --image docs/assets/examples/spaceship-snow/01_t2i_spaceship_snow.png --i2i-mode edit
```

### Qwen Image Edit 2509 / source / D

```sh
uv run mlxgen generate --model Qwen/Qwen-Image-Edit-2509 --prompt "Turn the source into a clean graphite pencil sketch of the same hard-landed spaceship scene. Preserve the spaceship identity, snowy canyon layout, tilted hull, bent landing struts, disturbed snow, debris, and smoke plume. White paper background, precise line art, no color fill, no blur." --width 432 --height 240 --steps 20 --seed 9203 --metadata --replace --output validation_outputs/edit_prepared_capability_2026_06_05/qwen_edit_2509_source_d_pencil_crash.png --image docs/assets/examples/spaceship-snow/01_t2i_spaceship_snow.png --i2i-mode edit
```

### Qwen Image Edit 2509 / source / E

```sh
uv run mlxgen generate --model Qwen/Qwen-Image-Edit-2509 --prompt "Use the first image as the pencil crash structure and the second image as the cinematic lighting and color reference. Produce one coherent image of the same hard-landed spaceship in the snow: graphite sketch lines with subtle cinematic blue-hour shading, stable canyon layout, solid spaceship, visible crash debris, no blur, no text." --width 432 --height 240 --steps 20 --seed 9204 --metadata --replace --output validation_outputs/edit_prepared_capability_2026_06_05/qwen_edit_2509_source_e_composition.png --image validation_outputs/edit_prepared_capability_2026_06_05/qwen_edit_2509_source_d_pencil_crash.png --image validation_outputs/edit_prepared_capability_2026_06_05/qwen_edit_2509_source_b_cinematic.png --i2i-mode multi-reference
```

### Qwen Image Edit 2509 / q8 prepared / B

```sh
uv run mlxgen generate --model AbstractFramework/qwen-image-edit-2509-8bit --prompt "Make this same spaceship in the snow look like polished cinematic science-fiction concept art at blue hour. Preserve the exact camera angle, ship position, snowy canyon, and overall layout. Sharpen hull panels and add cold blue shadows; no crash, no damage." --width 432 --height 240 --steps 20 --seed 9201 --metadata --replace --output validation_outputs/edit_prepared_capability_2026_06_05/qwen_edit_2509_8bit_b_cinematic.png --image docs/assets/examples/spaceship-snow/01_t2i_spaceship_snow.png --i2i-mode edit
```

### Qwen Image Edit 2509 / q8 prepared / C

```sh
uv run mlxgen generate --model AbstractFramework/qwen-image-edit-2509-8bit --prompt "Edit the source into the same spaceship after a hard landing in the snow. Preserve the same camera angle, spaceship identity, rear engines, canyon cliffs, and wide framing. The ship must remain solid and sharp, but show a tilted hull, bent landing struts, broken ice chunks, disturbed snow, a shallow scrape trail, and a thin smoke plume. No blur, no mesh, no dissolve." --width 432 --height 240 --steps 20 --seed 9202 --metadata --replace --output validation_outputs/edit_prepared_capability_2026_06_05/qwen_edit_2509_8bit_c_crash.png --image docs/assets/examples/spaceship-snow/01_t2i_spaceship_snow.png --i2i-mode edit
```

### Qwen Image Edit 2509 / q8 prepared / D

```sh
uv run mlxgen generate --model AbstractFramework/qwen-image-edit-2509-8bit --prompt "Turn the source into a clean graphite pencil sketch of the same hard-landed spaceship scene. Preserve the spaceship identity, snowy canyon layout, tilted hull, bent landing struts, disturbed snow, debris, and smoke plume. White paper background, precise line art, no color fill, no blur." --width 432 --height 240 --steps 20 --seed 9203 --metadata --replace --output validation_outputs/edit_prepared_capability_2026_06_05/qwen_edit_2509_8bit_d_pencil_crash.png --image docs/assets/examples/spaceship-snow/01_t2i_spaceship_snow.png --i2i-mode edit
```

### Qwen Image Edit 2509 / q8 prepared / E

```sh
uv run mlxgen generate --model AbstractFramework/qwen-image-edit-2509-8bit --prompt "Use the first image as the pencil crash structure and the second image as the cinematic lighting and color reference. Produce one coherent image of the same hard-landed spaceship in the snow: graphite sketch lines with subtle cinematic blue-hour shading, stable canyon layout, solid spaceship, visible crash debris, no blur, no text." --width 432 --height 240 --steps 20 --seed 9204 --metadata --replace --output validation_outputs/edit_prepared_capability_2026_06_05/qwen_edit_2509_8bit_e_composition.png --image validation_outputs/edit_prepared_capability_2026_06_05/qwen_edit_2509_8bit_d_pencil_crash.png --image validation_outputs/edit_prepared_capability_2026_06_05/qwen_edit_2509_8bit_b_cinematic.png --i2i-mode multi-reference
```

### Qwen Image Edit 2509 / q4 prepared / B

```sh
uv run mlxgen generate --model AbstractFramework/qwen-image-edit-2509-4bit --prompt "Make this same spaceship in the snow look like polished cinematic science-fiction concept art at blue hour. Preserve the exact camera angle, ship position, snowy canyon, and overall layout. Sharpen hull panels and add cold blue shadows; no crash, no damage." --width 432 --height 240 --steps 20 --seed 9201 --metadata --replace --output validation_outputs/i2i_standard_sequence_2026_06_05/qwen_edit_2509_4bit_b_cinematic.png --image docs/assets/examples/spaceship-snow/01_t2i_spaceship_snow.png --i2i-mode edit
```

### Qwen Image Edit 2509 / q4 prepared / C

```sh
uv run mlxgen generate --model AbstractFramework/qwen-image-edit-2509-4bit --prompt "Edit the source into the same spaceship after a hard landing in the snow. Preserve the same camera angle, spaceship identity, rear engines, canyon cliffs, and wide framing. The ship must remain solid and sharp, but show a tilted hull, bent landing struts, broken ice chunks, disturbed snow, a shallow scrape trail, and a thin smoke plume. No blur, no mesh, no dissolve." --width 432 --height 240 --steps 20 --seed 9202 --metadata --replace --output validation_outputs/i2i_standard_sequence_2026_06_05/qwen_edit_2509_4bit_c_crash.png --image docs/assets/examples/spaceship-snow/01_t2i_spaceship_snow.png --i2i-mode edit
```

### Qwen Image Edit 2509 / q4 prepared / D

```sh
uv run mlxgen generate --model AbstractFramework/qwen-image-edit-2509-4bit --prompt "Turn the source into a clean graphite pencil sketch of the same hard-landed spaceship scene. Preserve the spaceship identity, snowy canyon layout, tilted hull, bent landing struts, disturbed snow, debris, and smoke plume. White paper background, precise line art, no color fill, no blur." --width 432 --height 240 --steps 20 --seed 9203 --metadata --replace --output validation_outputs/i2i_standard_sequence_2026_06_05/qwen_edit_2509_4bit_d_pencil_crash.png --image docs/assets/examples/spaceship-snow/01_t2i_spaceship_snow.png --i2i-mode edit
```

### Qwen Image Edit 2509 / q4 prepared / E

```sh
uv run mlxgen generate --model AbstractFramework/qwen-image-edit-2509-4bit --prompt "Use the first image as the pencil crash structure and the second image as the cinematic lighting and color reference. Produce one coherent image of the same hard-landed spaceship in the snow: graphite sketch lines with subtle cinematic blue-hour shading, stable canyon layout, solid spaceship, visible crash debris, no blur, no text." --width 432 --height 240 --steps 20 --seed 9204 --metadata --replace --output validation_outputs/i2i_standard_sequence_2026_06_05/qwen_edit_2509_4bit_e_composition.png --image validation_outputs/i2i_standard_sequence_2026_06_05/qwen_edit_2509_4bit_d_pencil_crash.png --image validation_outputs/i2i_standard_sequence_2026_06_05/qwen_edit_2509_4bit_b_cinematic.png --i2i-mode multi-reference
```
