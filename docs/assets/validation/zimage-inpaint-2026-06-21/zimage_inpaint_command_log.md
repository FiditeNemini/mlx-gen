# Z-Image Turbo native inpaint proof commands

Accepted route:

- model: `AbstractFramework/z-image-turbo-8bit`
- baseline capability: `z-image.latent`
- new route capability: `z-image.inpaint`

## Same-prompt latent baseline

```sh
uv run mlxgen generate \
  --model AbstractFramework/z-image-turbo-8bit \
  --image docs/assets/examples/spaceship-snow/01_t2i_spaceship_snow.png \
  --image-strength 0.35 \
  --prompt "Keep the same silver spaceship, icy canyon, and sunrise lighting. Only inside the masked engine area, intensify both blue engines into brighter plasma thrusters, add dense blue glow and snow vapor around the thrusters, and preserve the rest of the image unchanged." \
  --width 768 \
  --height 432 \
  --steps 9 \
  --seed 4201 \
  --metadata \
  --replace \
  --output validation_outputs/masked_generation_routes_2026_06_21/generated/zimage_latent_engine_q8.png
```

## Native inpaint on the same prompt and seed

```sh
uv run mlxgen generate \
  --model AbstractFramework/z-image-turbo-8bit \
  --image docs/assets/examples/spaceship-snow/01_t2i_spaceship_snow.png \
  --mask-path validation_outputs/qwen_inpaint_2026_06_15/masks/source01_engine_mask.png \
  --prompt "Keep the same silver spaceship, icy canyon, and sunrise lighting. Only inside the masked engine area, intensify both blue engines into brighter plasma thrusters, add dense blue glow and snow vapor around the thrusters, and preserve the rest of the image unchanged." \
  --width 768 \
  --height 432 \
  --steps 9 \
  --seed 4201 \
  --metadata \
  --replace \
  --output validation_outputs/masked_generation_routes_2026_06_21/generated/zimage_inpaint_engine_q8.png
```
