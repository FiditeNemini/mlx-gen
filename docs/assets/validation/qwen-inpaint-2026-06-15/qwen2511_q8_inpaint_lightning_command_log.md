# Qwen Image Edit 2511 q8 masked edit proof commands

## Engine boost, regular 20-step q8

```sh
mlxgen generate \
  --model AbstractFramework/qwen-image-edit-2511-8bit \
  --image docs/assets/examples/spaceship-snow/01_t2i_spaceship_snow.png \
  --mask-path validation_outputs/qwen_inpaint_2026_06_15/masks/source01_engine_mask.png \
  --prompt "Keep the same silver spaceship, icy canyon, and sunrise lighting. Only inside the masked engine area, intensify both blue engines into brighter plasma thrusters, add dense blue glow and snow vapor around the thrusters, and preserve the rest of the image unchanged." \
  --negative "blurry, low quality, distorted, deformed, extra ship parts, changed camera angle, changed background, text, watermark" \
  --width 768 \
  --height 432 \
  --steps 20 \
  --guidance 4 \
  --seed 4201 \
  --metadata \
  --replace \
  --output validation_outputs/qwen_inpaint_2026_06_15/generated/source01_engine_base20_q8.png
```

## Engine boost, 4-step Lightning q8

```sh
mlxgen generate \
  --model AbstractFramework/qwen-image-edit-2511-8bit \
  --image docs/assets/examples/spaceship-snow/01_t2i_spaceship_snow.png \
  --mask-path validation_outputs/qwen_inpaint_2026_06_15/masks/source01_engine_mask.png \
  --prompt "Keep the same silver spaceship, icy canyon, and sunrise lighting. Only inside the masked engine area, intensify both blue engines into brighter plasma thrusters, add dense blue glow and snow vapor around the thrusters, and preserve the rest of the image unchanged." \
  --negative "blurry, low quality, distorted, deformed, extra ship parts, changed camera angle, changed background, text, watermark" \
  --width 768 \
  --height 432 \
  --steps 4 \
  --guidance 1 \
  --seed 4201 \
  --metadata \
  --replace \
  --output validation_outputs/qwen_inpaint_2026_06_15/generated/source01_engine_lightning_q8.png \
  --lora-paths lightx2v/Qwen-Image-Edit-2511-Lightning:Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors \
  --lora-scales 1
```

## Crash repair, regular 20-step q8

```sh
mlxgen generate \
  --model AbstractFramework/qwen-image-edit-2511-8bit \
  --image docs/assets/examples/spaceship-snow/03_i2i_crash_snow.png \
  --mask-path validation_outputs/qwen_inpaint_2026_06_15/masks/source03_repair_mask.png \
  --prompt "Keep the same crashed silver spaceship, icy canyon, debris, and snowfall outside the mask. Inside the masked hull area, repair the cockpit and front body with clean silver panels and intact windows, while preserving the original camera framing and environment." \
  --negative "blurry, low quality, distorted, deformed, extra ship parts, changed camera angle, changed background, text, watermark" \
  --width 768 \
  --height 432 \
  --steps 20 \
  --guidance 4 \
  --seed 4301 \
  --metadata \
  --replace \
  --output validation_outputs/qwen_inpaint_2026_06_15/generated/source03_repair_base20_q8.png
```

## Crash repair, 4-step Lightning q8

```sh
mlxgen generate \
  --model AbstractFramework/qwen-image-edit-2511-8bit \
  --image docs/assets/examples/spaceship-snow/03_i2i_crash_snow.png \
  --mask-path validation_outputs/qwen_inpaint_2026_06_15/masks/source03_repair_mask.png \
  --prompt "Keep the same crashed silver spaceship, icy canyon, debris, and snowfall outside the mask. Inside the masked hull area, repair the cockpit and front body with clean silver panels and intact windows, while preserving the original camera framing and environment." \
  --negative "blurry, low quality, distorted, deformed, extra ship parts, changed camera angle, changed background, text, watermark" \
  --width 768 \
  --height 432 \
  --steps 4 \
  --guidance 1 \
  --seed 4301 \
  --metadata \
  --replace \
  --output validation_outputs/qwen_inpaint_2026_06_15/generated/source03_repair_lightning_q8.png \
  --lora-paths lightx2v/Qwen-Image-Edit-2511-Lightning:Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors \
  --lora-scales 1
```

## Engine boost control, 4-step Lightning q8 without mask

```sh
mlxgen generate \
  --model AbstractFramework/qwen-image-edit-2511-8bit \
  --image docs/assets/examples/spaceship-snow/01_t2i_spaceship_snow.png \
  --prompt "Keep the same silver spaceship, icy canyon, and sunrise lighting. Only inside the masked engine area, intensify both blue engines into brighter plasma thrusters, add dense blue glow and snow vapor around the thrusters, and preserve the rest of the image unchanged." \
  --negative "blurry, low quality, distorted, deformed, extra ship parts, changed camera angle, changed background, text, watermark" \
  --width 768 \
  --height 432 \
  --steps 4 \
  --guidance 1 \
  --seed 4201 \
  --metadata \
  --replace \
  --output validation_outputs/qwen_inpaint_2026_06_15/generated/source01_engine_lightning_nomask_q8.png \
  --lora-paths lightx2v/Qwen-Image-Edit-2511-Lightning:Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors \
  --lora-scales 1
```

## Crash repair control, 4-step Lightning q8 without mask

```sh
mlxgen generate \
  --model AbstractFramework/qwen-image-edit-2511-8bit \
  --image docs/assets/examples/spaceship-snow/03_i2i_crash_snow.png \
  --prompt "Keep the same crashed silver spaceship, icy canyon, debris, and snowfall outside the mask. Inside the masked hull area, repair the cockpit and front body with clean silver panels and intact windows, while preserving the original camera framing and environment." \
  --negative "blurry, low quality, distorted, deformed, extra ship parts, changed camera angle, changed background, text, watermark" \
  --width 768 \
  --height 432 \
  --steps 4 \
  --guidance 1 \
  --seed 4301 \
  --metadata \
  --replace \
  --output validation_outputs/qwen_inpaint_2026_06_15/generated/source03_repair_lightning_nomask_q8.png \
  --lora-paths lightx2v/Qwen-Image-Edit-2511-Lightning:Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors \
  --lora-scales 1
```
