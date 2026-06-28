# LoRA Route Expansion Command Log

These are the exact commands for the current published rows in the June 22 route-expansion bundle.

## Download The Adapters

```sh
mlxgen download --model flymy-ai/qwen-image-realism-lora --all-files
mlxgen download --model prithivMLmods/Qwen-Image-Studio-Realism --all-files
mlxgen download --model lightx2v/Qwen-Image-Edit-2511-Lightning --all-files
mlxgen download --model fal/Qwen-Image-Edit-2511-Multiple-Angles-LoRA --all-files
mlxgen download --model reverentelusarca/ernie-image-elusarca-anime-style-lora --all-files
mlxgen download --model dx8152/Flux2-Klein-9B-Migration --all-files
mlxgen download --model fal/flux-2-klein-4B-outpaint-lora --all-files
```

## Base Qwen Image q8 Text LoRA

No LoRA baseline:

```sh
mlxgen generate \
  --model AbstractFramework/qwen-image-8bit \
  --prompt "Realism portrait of a young woman of African descent standing in a sunlit park, arms crossed, dramatic natural lighting, three-quarter view, delicate jewelry, loose shoulder-length curls, natural skin texture, environmental portrait photography." \
  --negative "cartoon, painting, low quality, deformed face, extra limbs, text, watermark" \
  --width 512 \
  --height 512 \
  --steps 24 \
  --guidance 5 \
  --seed 9981 \
  --metadata \
  --replace \
  --output validation_outputs/lora_route_expansion_2026_06_22/qwen_q8_text_realism_no_lora.png
```

With LoRA:

```sh
mlxgen generate \
  --model AbstractFramework/qwen-image-8bit \
  --prompt "Realism portrait of a young woman of African descent standing in a sunlit park, arms crossed, dramatic natural lighting, three-quarter view, delicate jewelry, loose shoulder-length curls, natural skin texture, environmental portrait photography." \
  --negative "cartoon, painting, low quality, deformed face, extra limbs, text, watermark" \
  --width 512 \
  --height 512 \
  --steps 24 \
  --guidance 5 \
  --seed 9981 \
  --metadata \
  --replace \
  --output validation_outputs/lora_route_expansion_2026_06_22/qwen_q8_text_realism_with_lora.png \
  --lora-paths flymy-ai/qwen-image-realism-lora:flymy_realism.safetensors \
  --lora-scales 1.0
```

## Base Qwen Image q8 Latent Img2Img LoRA

This row uses the latent source below:

- `docs/assets/validation/lora-route-expansion-2026-06-22/qwen_q8_latent_source_portrait_illustration.png`

No LoRA baseline:

```sh
mlxgen generate \
  --model AbstractFramework/qwen-image-8bit \
  --image docs/assets/validation/lora-route-expansion-2026-06-22/qwen_q8_latent_source_portrait_illustration.png \
  --i2i-mode latent \
  --image-strength 0.6 \
  --prompt "Studio Realism, photorealistic portrait of the same young woman of African descent standing in the same sunlit park with arms crossed, the same loose shoulder-length curls, the same pendant necklace, and the same sleeveless taupe dress. Preserve the same pose, framing, and background layout. Natural skin texture, realistic hair strands, subtle outdoor depth of field, no text." \
  --width 512 \
  --height 512 \
  --steps 24 \
  --guidance 5 \
  --seed 4421 \
  --metadata \
  --replace \
  --output validation_outputs/production_qwen_flux2_routes_2026_06_22/qwen_q8_latent_portrait_studio_cfg_auto_no_lora.png
```

With LoRA:

```sh
mlxgen generate \
  --model AbstractFramework/qwen-image-8bit \
  --image docs/assets/validation/lora-route-expansion-2026-06-22/qwen_q8_latent_source_portrait_illustration.png \
  --i2i-mode latent \
  --image-strength 0.6 \
  --prompt "Studio Realism, photorealistic portrait of the same young woman of African descent standing in the same sunlit park with arms crossed, the same loose shoulder-length curls, the same pendant necklace, and the same sleeveless taupe dress. Preserve the same pose, framing, and background layout. Natural skin texture, realistic hair strands, subtle outdoor depth of field, no text." \
  --width 512 \
  --height 512 \
  --steps 24 \
  --guidance 5 \
  --seed 4421 \
  --metadata \
  --replace \
  --output validation_outputs/production_qwen_flux2_routes_2026_06_22/qwen_q8_latent_portrait_studio_cfg_auto_with_lora.png \
  --lora-paths prithivMLmods/Qwen-Image-Studio-Realism:qwen-studio-realism.safetensors \
  --lora-scales 1.0
```

## Qwen Image Edit 2511 q8 Reframe LoRA

Prompt-matched Lightning-only baseline:

```sh
mlxgen generate \
  --model AbstractFramework/qwen-image-edit-2511-8bit \
  --image docs/assets/validation/reframe-outpaint-2026-06-08/source-b-cropped-starship.png \
  --reframe-padding "25%,80%,25%,60%" \
  --prompt "Use the expanded canvas to reveal the same spaceship as a back view low-angle wide shot. <sks> back view low-angle shot wide shot. Keep the same silver starship identity, snowy canyon environment, and coherent wide framing. Show the rear engines and tail from behind, keep the scene sharp, and do not add a second ship, text, or border." \
  --width 1040 \
  --height 368 \
  --steps 4 \
  --guidance 1 \
  --seed 9981 \
  --metadata \
  --replace \
  --output validation_outputs/production_qwen_flux2_routes_2026_06_22/qwen2511_q8_reframe_multiangle_promptmatched_no_lora.png \
  --lora-paths lightx2v/Qwen-Image-Edit-2511-Lightning:Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors \
  --lora-scales 1.0
```

Lightning plus multi-angle:

```sh
mlxgen generate \
  --model AbstractFramework/qwen-image-edit-2511-8bit \
  --image docs/assets/validation/reframe-outpaint-2026-06-08/source-b-cropped-starship.png \
  --reframe-padding "25%,80%,25%,60%" \
  --prompt "Use the expanded canvas to reveal the same spaceship as a back view low-angle wide shot. <sks> back view low-angle shot wide shot. Keep the same silver starship identity, snowy canyon environment, and coherent wide framing. Show the rear engines and tail from behind, keep the scene sharp, and do not add a second ship, text, or border." \
  --width 1040 \
  --height 368 \
  --steps 4 \
  --guidance 1 \
  --seed 9981 \
  --metadata \
  --replace \
  --output validation_outputs/qwen_flux2_route_completion_2026_06_22/qwen2511_q8_reframe_lightning_plus_multiangle.png \
  --lora-paths lightx2v/Qwen-Image-Edit-2511-Lightning:Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors fal/Qwen-Image-Edit-2511-Multiple-Angles-LoRA:qwen-image-edit-2511-multiple-angles-lora.safetensors \
  --lora-scales 1.0 0.9
```

## Qwen Image Edit 2511 q8 Outpaint LoRA

Prompt-matched Lightning-only baseline:

```sh
mlxgen generate \
  --model AbstractFramework/qwen-image-edit-2511-8bit \
  --image docs/assets/validation/reframe-outpaint-2026-06-08/source-b-cropped-starship.png \
  --outpaint-padding "5%,80%,5%,60%" \
  --prompt "Outpaint this cropped starship image into a back view low-angle wide shot of the same spacecraft in the snowy canyon. <sks> back view low-angle shot wide shot. Keep the same silver starship identity, snowy canyon environment, and coherent wide framing. Reveal the rear engines and tail in the new space, keep it sharp, and do not add a second ship, text, or border." \
  --width 1040 \
  --height 272 \
  --steps 4 \
  --guidance 1 \
  --seed 9982 \
  --metadata \
  --replace \
  --output validation_outputs/production_qwen_flux2_routes_2026_06_22/qwen2511_q8_outpaint_multiangle_promptmatched_no_lora.png \
  --lora-paths lightx2v/Qwen-Image-Edit-2511-Lightning:Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors \
  --lora-scales 1.0
```

Lightning plus multi-angle:

```sh
mlxgen generate \
  --model AbstractFramework/qwen-image-edit-2511-8bit \
  --image docs/assets/validation/reframe-outpaint-2026-06-08/source-b-cropped-starship.png \
  --outpaint-padding "5%,80%,5%,60%" \
  --prompt "Outpaint this cropped starship image into a back view low-angle wide shot of the same spacecraft in the snowy canyon. <sks> back view low-angle shot wide shot. Keep the same silver starship identity, snowy canyon environment, and coherent wide framing. Reveal the rear engines and tail in the new space, keep it sharp, and do not add a second ship, text, or border." \
  --width 1040 \
  --height 272 \
  --steps 4 \
  --guidance 1 \
  --seed 9982 \
  --metadata \
  --replace \
  --output validation_outputs/qwen_flux2_route_completion_2026_06_22/qwen2511_q8_outpaint_lightning_plus_multiangle.png \
  --lora-paths lightx2v/Qwen-Image-Edit-2511-Lightning:Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors fal/Qwen-Image-Edit-2511-Multiple-Angles-LoRA:qwen-image-edit-2511-multiple-angles-lora.safetensors \
  --lora-scales 1.0 0.9
```

## Qwen Image Edit 2511 q8 Multi-Reference LoRA

Prompt-matched Lightning-only baseline:

```sh
mlxgen generate \
  --model AbstractFramework/qwen-image-edit-2511-8bit \
  --image docs/assets/validation/qwen-edit-2511-parity-2026-06-06/qwen2511-source-pencil.png \
  --image docs/assets/validation/qwen-edit-2511-parity-2026-06-06/qwen2511-source-crash.png \
  --prompt "Use the first image as the graphite pencil sketch style reference and the second image as the hard-landing crash content reference. <sks> back view low-angle shot wide shot. Produce one coherent wide image of the same spaceship crashed in the snowy canyon from behind at a low camera angle: graphite pencil outlines on white paper, visible tilted hull, disturbed snow, broken ice chunks, scrape trail, and a thin smoke plume. Preserve the spaceship identity and canyon layout. No blur, no colored photo, no text." \
  --width 432 \
  --height 240 \
  --steps 4 \
  --guidance 1 \
  --seed 9971 \
  --metadata \
  --replace \
  --output validation_outputs/production_qwen_flux2_routes_2026_06_22/qwen2511_q8_multi_reference_multiangle_promptmatched_no_lora.png \
  --lora-paths lightx2v/Qwen-Image-Edit-2511-Lightning:Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors \
  --lora-scales 1.0
```

Lightning plus multi-angle:

```sh
mlxgen generate \
  --model AbstractFramework/qwen-image-edit-2511-8bit \
  --image docs/assets/validation/qwen-edit-2511-parity-2026-06-06/qwen2511-source-pencil.png \
  --image docs/assets/validation/qwen-edit-2511-parity-2026-06-06/qwen2511-source-crash.png \
  --prompt "Use the first image as the graphite pencil sketch style reference and the second image as the hard-landing crash content reference. <sks> back view low-angle shot wide shot. Produce one coherent wide image of the same spaceship crashed in the snowy canyon from behind at a low camera angle: graphite pencil outlines on white paper, visible tilted hull, disturbed snow, broken ice chunks, scrape trail, and a thin smoke plume. Preserve the spaceship identity and canyon layout. No blur, no colored photo, no text." \
  --width 432 \
  --height 240 \
  --steps 4 \
  --guidance 1 \
  --seed 9971 \
  --metadata \
  --replace \
  --output validation_outputs/qwen_flux2_route_completion_2026_06_22/qwen2511_q8_multi_reference_lightning_plus_multiangle.png \
  --lora-paths lightx2v/Qwen-Image-Edit-2511-Lightning:Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors fal/Qwen-Image-Edit-2511-Multiple-Angles-LoRA:qwen-image-edit-2511-multiple-angles-lora.safetensors \
  --lora-scales 1.0 0.9
```

## ERNIE Image Turbo q8 Latent Img2Img LoRA

This row uses the same base-Qwen no-LoRA portrait source.

No LoRA baseline:

```sh
mlxgen generate \
  --model AbstractFramework/ernie-image-turbo-8bit \
  --image validation_outputs/lora_route_expansion_2026_06_22/qwen_q8_text_realism_no_lora.png \
  --i2i-mode latent \
  --image-strength 0.75 \
  --prompt "elusarca anime style, portrait of a young woman of African descent standing in a sunlit park, arms crossed, delicate jewelry, loose shoulder-length curls, preserve the same pose and park layout, polished anime illustration, crisp linework, soft luminous skin, cinematic outdoor light." \
  --negative "blurry, deformed face, extra limbs, text, watermark" \
  --width 512 \
  --height 512 \
  --steps 8 \
  --guidance 1 \
  --seed 9981 \
  --metadata \
  --replace \
  --output validation_outputs/lora_route_expansion_2026_06_22/ernie_turbo_q8_latent_portrait75_no_lora.png
```

With LoRA:

```sh
mlxgen generate \
  --model AbstractFramework/ernie-image-turbo-8bit \
  --image validation_outputs/lora_route_expansion_2026_06_22/qwen_q8_text_realism_no_lora.png \
  --i2i-mode latent \
  --image-strength 0.75 \
  --prompt "elusarca anime style, portrait of a young woman of African descent standing in a sunlit park, arms crossed, delicate jewelry, loose shoulder-length curls, preserve the same pose and park layout, polished anime illustration, crisp linework, soft luminous skin, cinematic outdoor light." \
  --negative "blurry, deformed face, extra limbs, text, watermark" \
  --width 512 \
  --height 512 \
  --steps 8 \
  --guidance 1 \
  --seed 9981 \
  --metadata \
  --replace \
  --output validation_outputs/lora_route_expansion_2026_06_22/ernie_turbo_q8_latent_portrait75_with_lora.png \
  --lora-paths reverentelusarca/ernie-image-elusarca-anime-style-lora:ernie-anime-v1.safetensors \
  --lora-scales 0.9
```

## FLUX.2 Klein 9B q8 Multi-Reference LoRA

No LoRA baseline:

```sh
mlxgen generate \
  --model AbstractFramework/flux.2-klein-9b-8bit \
  --image docs/assets/validation/i2i-edit-5x4-2026-06-05/reference-inputs/flux2_klein_9b_8bit_d_pencil_crash.png \
  --image docs/assets/validation/i2i-edit-5x4-2026-06-05/reference-inputs/flux2_klein_9b_8bit_b_cinematic.png \
  --prompt "Use the two references to build one coherent scene: keep the hard-landed spaceship identity and snowy canyon composition from the first image, and bring over the lighting, cleaner atmosphere, and richer color treatment from the second image. Preserve one ship, one canyon, no text." \
  --width 432 \
  --height 240 \
  --steps 20 \
  --guidance 1 \
  --seed 8661 \
  --metadata \
  --replace \
  --output validation_outputs/production_flux2_routes_2026_06_22/flux2_klein9b_q8_multiref_no_lora.png
```

With LoRA:

```sh
mlxgen generate \
  --model AbstractFramework/flux.2-klein-9b-8bit \
  --image docs/assets/validation/i2i-edit-5x4-2026-06-05/reference-inputs/flux2_klein_9b_8bit_d_pencil_crash.png \
  --image docs/assets/validation/i2i-edit-5x4-2026-06-05/reference-inputs/flux2_klein_9b_8bit_b_cinematic.png \
  --prompt "Use the two references to build one coherent scene: keep the hard-landed spaceship identity and snowy canyon composition from the first image, and bring over the lighting, cleaner atmosphere, and richer color treatment from the second image. Preserve one ship, one canyon, no text." \
  --width 432 \
  --height 240 \
  --steps 20 \
  --guidance 1 \
  --seed 8661 \
  --metadata \
  --replace \
  --output validation_outputs/production_flux2_routes_2026_06_22/flux2_klein9b_q8_multiref_migration.png \
  --lora-paths dx8152/Flux2-Klein-9B-Migration:Klein-Migration.safetensors \
  --lora-scales 0.8
```

## FLUX.2 Klein Base 4B q8 Outpaint LoRA

The base route already supports outpaint without a LoRA. This exact row uses the same generated
green canvas and seed to isolate the dedicated outpaint adapter.

No LoRA baseline:

```sh
mlxgen generate \
  --model AbstractFramework/flux.2-klein-base-4b-8bit \
  --image docs/assets/validation/reframe-outpaint-2026-06-08/source-b-cropped-starship.png \
  --outpaint-padding "5%,80%,5%,60%" \
  --prompt "Fill the green spaces according to the image" \
  --width 1040 \
  --height 272 \
  --steps 20 \
  --guidance 4 \
  --seed 8612 \
  --metadata \
  --replace \
  --output validation_outputs/production_flux2_routes_2026_06_22/flux2_klein_base4b_q8_outpaint_route_no_lora.png
```

With LoRA:

```sh
mlxgen generate \
  --model AbstractFramework/flux.2-klein-base-4b-8bit \
  --image docs/assets/validation/reframe-outpaint-2026-06-08/source-b-cropped-starship.png \
  --outpaint-padding "5%,80%,5%,60%" \
  --prompt "Fill the green spaces according to the image" \
  --width 1040 \
  --height 272 \
  --steps 20 \
  --guidance 4 \
  --seed 8612 \
  --metadata \
  --replace \
  --output validation_outputs/production_flux2_routes_2026_06_22/flux2_klein_base4b_q8_outpaint_route_with_lora.png \
  --lora-paths fal/flux-2-klein-4B-outpaint-lora:flux-outpaint-lora.safetensors \
  --lora-scales 1.0
```
