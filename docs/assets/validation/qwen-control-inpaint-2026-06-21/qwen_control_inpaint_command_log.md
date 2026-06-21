# Qwen base control-inpaint proof commands

Accepted route:

- model: `AbstractFramework/qwen-image-8bit`
- capability: `qwen.control-inpaint`
- exact sidecar injected by `mlxgen generate`:
  `InstantX/Qwen-Image-ControlNet-Inpainting:diffusion_pytorch_model.safetensors`
- fast adapter:
  `lightx2v/Qwen-Image-Lightning:Qwen-Image-Lightning-4steps-V2.0-bf16.safetensors`

## Engine thruster, same-seed base-Qwen control-inpaint

```sh
uv run mlxgen generate \
  --model AbstractFramework/qwen-image-8bit \
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
  --lora-paths lightx2v/Qwen-Image-Lightning:Qwen-Image-Lightning-4steps-V2.0-bf16.safetensors \
  --lora-scales 1 \
  --output validation_outputs/masked_generation_routes_2026_06_21/generated/qwen_control_inpaint_engine_lightning_q8.png
```

## Hull repair, same-seed base-Qwen control-inpaint

```sh
uv run mlxgen generate \
  --model AbstractFramework/qwen-image-8bit \
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
  --lora-paths lightx2v/Qwen-Image-Lightning:Qwen-Image-Lightning-4steps-V2.0-bf16.safetensors \
  --lora-scales 1 \
  --output validation_outputs/masked_generation_routes_2026_06_21/generated/qwen_control_inpaint_repair_lightning_q8.png
```

Comparison column used in the published contact sheet:

- accepted masked-edit route:
  [qwen2511_q8_inpaint_lightning_command_log.md](../qwen-inpaint-2026-06-15/qwen2511_q8_inpaint_lightning_command_log.md)
