# FLUX.2 Klein Base Starship Validation Commands

Canonical source: `docs/assets/validation/reframe-outpaint-2026-06-08/source-b-cropped-starship.png`

T2I prompt: `A polished retro-futuristic silver starship resting in a snowy canyon, cinematic winter light, sharp metallic panels, icy cliffs, ultra-detailed concept art.`

## FLUX.2 Klein Base 9B / text-to-image

```sh
uv run mlxgen generate --model black-forest-labs/FLUX.2-klein-base-9B --prompt "A polished retro-futuristic silver starship resting in a snowy canyon, cinematic winter light, sharp metallic panels, icy cliffs, ultra-detailed concept art." --width 432 --height 240 --steps 20 --guidance 1.5 --seed 8610 --metadata --replace --output validation_outputs/flux2_klein_base_starship_2026_06_10/base9b_source_t2i_starship.png
```

## FLUX.2 Klein Base 9B / Latent aurora

```sh
uv run mlxgen generate --model black-forest-labs/FLUX.2-klein-base-9B --image docs/assets/validation/reframe-outpaint-2026-06-08/source-b-cropped-starship.png --i2i-mode latent --image-strength 0.75 --prompt "Transform this exact close-up starship crop into a clearly darker polar night scene with a visible teal aurora over the peaks. Preserve the exact crop, camera angle, fuselage shape, engines, snow field, and ice cliffs. Make the sky distinctly navy, deepen the snow shadows, and add obvious cyan aurora reflections across the silver metal. No damage, no extra ships, no text." --width 432 --height 240 --steps 20 --guidance 3.0 --seed 8611 --metadata --replace --output validation_outputs/flux2_klein_base_starship_2026_06_10/base9b_source_b_latent_dusk.png
```

## FLUX.2 Klein Base 9B / Damage edit

```sh
uv run mlxgen generate --model black-forest-labs/FLUX.2-klein-base-9B --image docs/assets/validation/reframe-outpaint-2026-06-08/source-b-cropped-starship.png --i2i-mode edit --prompt "Edit this same close-up starship into a hard-landed damaged version. Preserve the exact crop, camera angle, fuselage shape, cockpit, engine positions, snow field, and ice cliffs. Add scraped metal, bent panels, soot near the intakes, a thin smoke plume, disturbed snow, and a shallow impact groove. Keep the ship sharp and coherent. No extra ships, no blur, no text." --width 432 --height 240 --steps 20 --guidance 1.5 --seed 8612 --metadata --replace --output validation_outputs/flux2_klein_base_starship_2026_06_10/base9b_source_c_damage.png
```

## FLUX.2 Klein Base 9B / Sketch edit

```sh
uv run mlxgen generate --model black-forest-labs/FLUX.2-klein-base-9B --image docs/assets/validation/reframe-outpaint-2026-06-08/source-b-cropped-starship.png --i2i-mode edit --prompt "Turn this same close-up starship scene into a clean graphite pencil sketch. Preserve the exact crop, camera angle, fuselage shape, cockpit, engines, snow field, and ice cliffs. Use white paper, precise line art, subtle shading, no color fill, no blur, no text." --width 432 --height 240 --steps 20 --guidance 1.5 --seed 8613 --metadata --replace --output validation_outputs/flux2_klein_base_starship_2026_06_10/base9b_source_d_sketch.png
```

## FLUX.2 Klein Base 9B / Multi-reference

```sh
uv run mlxgen generate --model black-forest-labs/FLUX.2-klein-base-9B --image validation_outputs/flux2_klein_base_starship_2026_06_10/base9b_source_d_sketch.png --image validation_outputs/flux2_klein_base_starship_2026_06_10/base9b_source_b_latent_dusk.png --i2i-mode multi-reference --prompt "Use the first image as the structural line-art reference and the second image as the lighting and material reference. Produce one coherent close-up of the same starship scene: graphite line art with cool aurora metallic reflections and darker polar-night shading, the same crop, same fuselage and engines, same snow field and ice cliffs, no extra ships, no text." --width 432 --height 240 --steps 20 --guidance 1.5 --seed 8614 --metadata --replace --output validation_outputs/flux2_klein_base_starship_2026_06_10/base9b_source_e_multiref.png
```

## FLUX.2 Klein Base 9B / Strict outpaint

```sh
uv run mlxgen generate --model black-forest-labs/FLUX.2-klein-base-9B --image docs/assets/validation/reframe-outpaint-2026-06-08/source-b-cropped-starship.png --outpaint-padding "5%,80%,5%,60%" --prompt "Outpaint this close cropped starship image into a much wider realistic shot of the full spacecraft in the snowy canyon. Keep the existing central spacecraft surface consistent, and complete the missing nose, full hull, tail, engines, snow field, and ice cliffs in the newly added space. The entire ship must fit inside the final wide frame with empty snow visible around it. Preserve the same lighting and camera angle. No text, no frame, no border, no duplicate ship." --steps 20 --guidance 4 --seed 8612 --metadata --replace --output validation_outputs/flux2_base_outpaint_2026_06_10/flux2_base9b_source_starship_outpaint_seed8612_cli_final.png
```

## FLUX.2 Klein Base 4B / text-to-image

```sh
uv run mlxgen generate --model black-forest-labs/FLUX.2-klein-base-4B --prompt "A polished retro-futuristic silver starship resting in a snowy canyon, cinematic winter light, sharp metallic panels, icy cliffs, ultra-detailed concept art." --width 432 --height 240 --steps 20 --guidance 1.5 --seed 8610 --metadata --replace --output validation_outputs/flux2_klein_base_starship_2026_06_10/base4b_source_t2i_starship.png
```

## FLUX.2 Klein Base 4B / Latent aurora

```sh
uv run mlxgen generate --model black-forest-labs/FLUX.2-klein-base-4B --image docs/assets/validation/reframe-outpaint-2026-06-08/source-b-cropped-starship.png --i2i-mode latent --image-strength 0.75 --prompt "Transform this exact close-up starship crop into a clearly darker polar night scene with a visible teal aurora over the peaks. Preserve the exact crop, camera angle, fuselage shape, engines, snow field, and ice cliffs. Make the sky distinctly navy, deepen the snow shadows, and add obvious cyan aurora reflections across the silver metal. No damage, no extra ships, no text." --width 432 --height 240 --steps 20 --guidance 3.0 --seed 8611 --metadata --replace --output validation_outputs/flux2_klein_base_starship_2026_06_10/base4b_source_b_latent_dusk.png
```

## FLUX.2 Klein Base 4B / Damage edit

```sh
uv run mlxgen generate --model black-forest-labs/FLUX.2-klein-base-4B --image docs/assets/validation/reframe-outpaint-2026-06-08/source-b-cropped-starship.png --i2i-mode edit --prompt "Edit this same close-up starship into a hard-landed damaged version. Preserve the exact crop, camera angle, fuselage shape, cockpit, engine positions, snow field, and ice cliffs. Add scraped metal, bent panels, soot near the intakes, a thin smoke plume, disturbed snow, and a shallow impact groove. Keep the ship sharp and coherent. No extra ships, no blur, no text." --width 432 --height 240 --steps 20 --guidance 1.5 --seed 8612 --metadata --replace --output validation_outputs/flux2_klein_base_starship_2026_06_10/base4b_source_c_damage.png
```

## FLUX.2 Klein Base 4B / Sketch edit

```sh
uv run mlxgen generate --model black-forest-labs/FLUX.2-klein-base-4B --image docs/assets/validation/reframe-outpaint-2026-06-08/source-b-cropped-starship.png --i2i-mode edit --prompt "Turn this same close-up starship scene into a clean graphite pencil sketch. Preserve the exact crop, camera angle, fuselage shape, cockpit, engines, snow field, and ice cliffs. Use white paper, precise line art, subtle shading, no color fill, no blur, no text." --width 432 --height 240 --steps 20 --guidance 1.5 --seed 8613 --metadata --replace --output validation_outputs/flux2_klein_base_starship_2026_06_10/base4b_source_d_sketch.png
```

## FLUX.2 Klein Base 4B / Multi-reference

```sh
uv run mlxgen generate --model black-forest-labs/FLUX.2-klein-base-4B --image validation_outputs/flux2_klein_base_starship_2026_06_10/base4b_source_d_sketch.png --image docs/assets/validation/reframe-outpaint-2026-06-08/source-b-cropped-starship.png --i2i-mode multi-reference --prompt "Use the first image as the structural graphite sketch reference and the second image as the metallic material and lighting reference. Produce one coherent close-up of the same starship scene. Preserve the exact crop including the nose, cockpit edge, front engine, snow field, and ice cliffs. Keep clean graphite lines with subtle metallic shading. No extra ships, no text." --width 432 --height 240 --steps 20 --guidance 2.0 --seed 8614 --metadata --replace --output validation_outputs/flux2_klein_base_starship_2026_06_10/base4b_source_e_multiref.png
```

## FLUX.2 Klein Base 4B / Strict outpaint

```sh
uv run mlxgen generate --model black-forest-labs/FLUX.2-klein-base-4B --image docs/assets/validation/reframe-outpaint-2026-06-08/source-b-cropped-starship.png --outpaint-padding "5%,80%,5%,60%" --prompt "Outpaint this close cropped starship image into a much wider realistic shot of the full spacecraft in the snowy canyon. Keep the existing central spacecraft surface consistent, and complete the missing nose, full hull, tail, engines, snow field, and ice cliffs in the newly added space. The entire ship must fit inside the final wide frame with empty snow visible around it. Preserve the same lighting and camera angle. No text, no frame, no border, no duplicate ship." --steps 20 --guidance 4 --seed 8612 --metadata --replace --output validation_outputs/flux2_klein_base_starship_2026_06_10/base4b_source_f_outpaint.png
```
