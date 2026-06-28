# Z-Image Turbo q8 latent LoRA command log

## Download the adapter

```sh
mlxgen download --model ostris/z_image_turbo_childrens_drawings --all-files
```

## Exact accepted row

Source image:

- `docs/assets/examples/spaceship-snow/01_t2i_spaceship_snow.png`

No LoRA baseline:

```sh
mlxgen generate \
  --model AbstractFramework/z-image-turbo-8bit \
  --image docs/assets/examples/spaceship-snow/01_t2i_spaceship_snow.png \
  --i2i-mode latent \
  --image-strength 0.35 \
  --prompt "Turn this same spaceship in the snow into a childs wax-crayon drawing on white paper. Preserve the exact camera angle, ship position, snowy canyon layout, and single ship silhouette. Use thick uneven crayon lines, simple childlike shapes, and flat hand-colored fills." \
  --width 432 \
  --height 240 \
  --steps 20 \
  --seed 9201 \
  --metadata \
  --replace \
  --output validation_outputs/zimage_latent_lora_2026_06_24/zimage_q8_latent_childdraw_no_lora.png
```

With LoRA:

```sh
mlxgen generate \
  --model AbstractFramework/z-image-turbo-8bit \
  --image docs/assets/examples/spaceship-snow/01_t2i_spaceship_snow.png \
  --i2i-mode latent \
  --image-strength 0.35 \
  --prompt "Turn this same spaceship in the snow into a childs wax-crayon drawing on white paper. Preserve the exact camera angle, ship position, snowy canyon layout, and single ship silhouette. Use thick uneven crayon lines, simple childlike shapes, and flat hand-colored fills." \
  --width 432 \
  --height 240 \
  --steps 20 \
  --seed 9201 \
  --metadata \
  --replace \
  --output validation_outputs/zimage_latent_lora_2026_06_24/zimage_q8_latent_childdraw_with_lora.png \
  --lora-paths ostris/z_image_turbo_childrens_drawings:z_image_turbo_childrens_drawings.safetensors \
  --lora-scales 1.0
```
