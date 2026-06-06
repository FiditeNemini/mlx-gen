# Regular Qwen Image Edit Validation Commands

These are the exact commands used for the published `Qwen/Qwen-Image-Edit`,
`AbstractFramework/qwen-image-edit-8bit`, and `AbstractFramework/qwen-image-edit-4bit` proof sheet.

## Qwen/Qwen-Image-Edit / Pencil Sketch

```sh
uv run mlxgen generate \
  --model Qwen/Qwen-Image-Edit \
  --image docs/assets/examples/spaceship-snow/01_t2i_spaceship_snow.png \
  --prompt "Convert the source image into a clean graphite pencil sketch on white paper. Preserve the same wide camera framing, the same spaceship shape, and the icy canyon background. Use thin gray pencil outlines with light shading only. The final image must look like a hand drawn pencil sketch, not a blurred photo." \
  --negative "color, yellow grid lines, ruler lines, crop, close-up, blur, paint, watercolor, abstract blocks, text, watermark" \
  --width 768 \
  --height 432 \
  --steps 30 \
  --guidance 4 \
  --scheduler flow_match_euler_discrete \
  --seed 9501 \
  --metadata \
  --output validation_outputs/qwen_image_edit_base_repaired_2026_06_06/qwen_image_edit_base_vl1024_sketch_seed9501.png
```

## Qwen/Qwen-Image-Edit / Crash Edit

```sh
uv run mlxgen generate \
  --model Qwen/Qwen-Image-Edit \
  --image docs/assets/examples/spaceship-snow/01_t2i_spaceship_snow.png \
  --prompt "Edit the source image into a clear crash-landed spaceship scene at dusk. Preserve the same wide camera framing and keep the full spacecraft visible. Make the ship visibly crashed: nose buried in snow, hull tilted and damaged, one landing leg broken, metal debris scattered around the ship, a long dark scrape trench in the snow behind it, snow thrown up around the impact, and a thick gray smoke plume rising from the rear engines. Change the sky to warm orange dusk with blue snow shadows. Do not crop or zoom the ship." \
  --negative "cropped ship, close-up, intact landing, clean undamaged ship, no debris, no smoke, daylight, yellow grid lines, abstract blur, text, watermark" \
  --width 768 \
  --height 432 \
  --steps 30 \
  --guidance 4 \
  --scheduler flow_match_euler_discrete \
  --seed 9401 \
  --metadata \
  --output validation_outputs/qwen_image_edit_base_repaired_2026_06_06/qwen_image_edit_base_vl1024_crash_seed9401.png
```

## AbstractFramework/qwen-image-edit-8bit / Pencil Sketch

```sh
uv run mlxgen generate \
  --model AbstractFramework/qwen-image-edit-8bit \
  --image docs/assets/examples/spaceship-snow/01_t2i_spaceship_snow.png \
  --prompt "Convert the source image into a clean graphite pencil sketch on white paper. Preserve the same wide camera framing, the same spaceship shape, and the icy canyon background. Use thin gray pencil outlines with light shading only. The final image must look like a hand drawn pencil sketch, not a blurred photo." \
  --negative "color, yellow grid lines, ruler lines, crop, close-up, blur, paint, watercolor, abstract blocks, text, watermark" \
  --width 768 \
  --height 432 \
  --steps 30 \
  --guidance 4 \
  --scheduler flow_match_euler_discrete \
  --seed 9501 \
  --metadata \
  --output validation_outputs/qwen_image_edit_base_repaired_2026_06_06/qwen_image_edit_q8_vl1024_sketch_seed9501.png
```

## AbstractFramework/qwen-image-edit-8bit / Crash Edit

```sh
uv run mlxgen generate \
  --model AbstractFramework/qwen-image-edit-8bit \
  --image docs/assets/examples/spaceship-snow/01_t2i_spaceship_snow.png \
  --prompt "Edit the source image into a clear crash-landed spaceship scene at dusk. Preserve the same wide camera framing and keep the full spacecraft visible. Make the ship visibly crashed: nose buried in snow, hull tilted and damaged, one landing leg broken, metal debris scattered around the ship, a long dark scrape trench in the snow behind it, snow thrown up around the impact, and a thick gray smoke plume rising from the rear engines. Change the sky to warm orange dusk with blue snow shadows. Do not crop or zoom the ship." \
  --negative "cropped ship, close-up, intact landing, clean undamaged ship, no debris, no smoke, daylight, yellow grid lines, abstract blur, text, watermark" \
  --width 768 \
  --height 432 \
  --steps 30 \
  --guidance 4 \
  --scheduler flow_match_euler_discrete \
  --seed 9401 \
  --metadata \
  --output validation_outputs/qwen_image_edit_base_repaired_2026_06_06/qwen_image_edit_q8_vl1024_crash_seed9401.png
```

## AbstractFramework/qwen-image-edit-4bit / Pencil Sketch

```sh
uv run mlxgen generate \
  --model AbstractFramework/qwen-image-edit-4bit \
  --image docs/assets/examples/spaceship-snow/01_t2i_spaceship_snow.png \
  --prompt "Convert the source image into a clean graphite pencil sketch on white paper. Preserve the same wide camera framing, the same spaceship shape, and the icy canyon background. Use thin gray pencil outlines with light shading only. The final image must look like a hand drawn pencil sketch, not a blurred photo." \
  --negative "color, yellow grid lines, ruler lines, crop, close-up, blur, paint, watercolor, abstract blocks, text, watermark" \
  --width 768 \
  --height 432 \
  --steps 30 \
  --guidance 4 \
  --scheduler flow_match_euler_discrete \
  --seed 9501 \
  --metadata \
  --output validation_outputs/qwen_image_edit_base_repaired_2026_06_06/qwen_image_edit_q4_vl1024_sketch_seed9501.png
```

## AbstractFramework/qwen-image-edit-4bit / Crash Edit

```sh
uv run mlxgen generate \
  --model AbstractFramework/qwen-image-edit-4bit \
  --image docs/assets/examples/spaceship-snow/01_t2i_spaceship_snow.png \
  --prompt "Edit the source image into a clear crash-landed spaceship scene at dusk. Preserve the same wide camera framing and keep the full spacecraft visible. Make the ship visibly crashed: nose buried in snow, hull tilted and damaged, one landing leg broken, metal debris scattered around the ship, a long dark scrape trench in the snow behind it, snow thrown up around the impact, and a thick gray smoke plume rising from the rear engines. Change the sky to warm orange dusk with blue snow shadows. Do not crop or zoom the ship." \
  --negative "cropped ship, close-up, intact landing, clean undamaged ship, no debris, no smoke, daylight, yellow grid lines, abstract blur, text, watermark" \
  --width 768 \
  --height 432 \
  --steps 30 \
  --guidance 4 \
  --scheduler flow_match_euler_discrete \
  --seed 9401 \
  --metadata \
  --output validation_outputs/qwen_image_edit_base_repaired_2026_06_06/qwen_image_edit_q4_vl1024_crash_seed9401.png
```
