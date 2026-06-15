# Qwen Image q8 structured control proof

Exact files used for the accepted proof:

```sh
hf download InstantX/Qwen-Image-ControlNet-Union \
  diffusion_pytorch_model.safetensors \
  config.json \
  conds/canny.png \
  conds/pose.png \
  README.md

hf download lightx2v/Qwen-Image-Lightning \
  Qwen-Image-Lightning-4steps-V2.0-bf16.safetensors \
  README.md
```

Canny control:

```sh
mlxgen generate \
  --model AbstractFramework/qwen-image-8bit \
  --prompt "Aesthetics art, traditional asian pagoda, elaborate golden accents, sky blue and white color palette, swirling cloud pattern, digital illustration, east asian architecture, ornamental rooftop, intricate detailing on building, cultural representation." \
  --negative "blurry, low quality, distorted, deformed, text, watermark, ugly" \
  --width 576 \
  --height 864 \
  --steps 4 \
  --guidance 1 \
  --seed 5802 \
  --controlnet-image-path ./conds/canny.png \
  --controlnet-strength 0.9 \
  --lora-paths lightx2v/Qwen-Image-Lightning:Qwen-Image-Lightning-4steps-V2.0-bf16.safetensors \
  --lora-scales 1 \
  --output instantx_canny_control.png
```

Canny no-control baseline:

```sh
mlxgen generate \
  --model AbstractFramework/qwen-image-8bit \
  --prompt "Aesthetics art, traditional asian pagoda, elaborate golden accents, sky blue and white color palette, swirling cloud pattern, digital illustration, east asian architecture, ornamental rooftop, intricate detailing on building, cultural representation." \
  --negative "blurry, low quality, distorted, deformed, text, watermark, ugly" \
  --width 576 \
  --height 864 \
  --steps 4 \
  --guidance 1 \
  --seed 5802 \
  --lora-paths lightx2v/Qwen-Image-Lightning:Qwen-Image-Lightning-4steps-V2.0-bf16.safetensors \
  --lora-scales 1 \
  --output instantx_canny_no_control.png
```

Pose control:

```sh
mlxgen generate \
  --model AbstractFramework/qwen-image-8bit \
  --prompt "Photograph of a young man with light brown hair and a beard, wearing a beige flat cap, black leather jacket, gray shirt, brown pants, and white sneakers. He's sitting on a concrete ledge in front of a large circular window, with a cityscape reflected in the glass. The wall is cream-colored, and the sky is clear blue. His shadow is cast on the wall." \
  --negative "blurry, low quality, distorted, deformed, extra limbs, extra fingers, text, watermark, ugly" \
  --width 608 \
  --height 768 \
  --steps 4 \
  --guidance 1 \
  --seed 5803 \
  --controlnet-image-path ./conds/pose.png \
  --controlnet-strength 0.9 \
  --lora-paths lightx2v/Qwen-Image-Lightning:Qwen-Image-Lightning-4steps-V2.0-bf16.safetensors \
  --lora-scales 1 \
  --output instantx_pose_control.png
```

Pose no-control baseline:

```sh
mlxgen generate \
  --model AbstractFramework/qwen-image-8bit \
  --prompt "Photograph of a young man with light brown hair and a beard, wearing a beige flat cap, black leather jacket, gray shirt, brown pants, and white sneakers. He's sitting on a concrete ledge in front of a large circular window, with a cityscape reflected in the glass. The wall is cream-colored, and the sky is clear blue. His shadow is cast on the wall." \
  --negative "blurry, low quality, distorted, deformed, extra limbs, extra fingers, text, watermark, ugly" \
  --width 608 \
  --height 768 \
  --steps 4 \
  --guidance 1 \
  --seed 5803 \
  --lora-paths lightx2v/Qwen-Image-Lightning:Qwen-Image-Lightning-4steps-V2.0-bf16.safetensors \
  --lora-scales 1 \
  --output instantx_pose_no_control.png
```
