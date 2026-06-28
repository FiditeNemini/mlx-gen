# Z-Image Turbo q8 latent LoRA proof

Accepted proof surface:

- route: `mlxgen generate --model AbstractFramework/z-image-turbo-8bit --image ... --i2i-mode latent`
- resolved capability: `z-image.latent`
- exact adapter: `ostris/z_image_turbo_childrens_drawings:z_image_turbo_childrens_drawings.safetensors`
- source image: [spaceship snow source](../../../assets/examples/spaceship-snow/01_t2i_spaceship_snow.png)

Published artifacts:

- [contact sheet](zimage_q8_latent_childdraw_contact_sheet.png)
- [stats](zimage_latent_lora_stats_m5max.json)
- [commands](zimage_latent_lora_command_log.md)
- [baseline output](../../../../validation_outputs/zimage_latent_lora_2026_06_24/zimage_q8_latent_childdraw_no_lora.png)
- [with LoRA output](../../../../validation_outputs/zimage_latent_lora_2026_06_24/zimage_q8_latent_childdraw_with_lora.png)

What the sheet proves:

- the exact `AbstractFramework/z-image-turbo-8bit` latent img2img row accepts and applies a Z-Image Turbo LoRA cleanly;
- the accepted row keeps the same spaceship placement, snowy canyon layout, and overall silhouette from the source image;
- the adapter adds a clear hand-drawn children's-crayon treatment instead of acting like a no-op.

Measured on an M5 Max:

- latent baseline: `98.90s` generation
- latent + LoRA: `84.14s` generation

Direct review outcome:

- accepted for the exact q8 latent img2img row above
- public claim stays narrow: this validates the documented package, route, and adapter shown in this bundle
