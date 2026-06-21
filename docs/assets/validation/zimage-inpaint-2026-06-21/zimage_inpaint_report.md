# Z-Image Turbo native inpaint proof

Accepted proof surface:

- route: `mlxgen generate --model AbstractFramework/z-image-turbo-8bit --image ... --mask-path ...`
- resolved capability: `z-image.inpaint`
- comparison baseline: same model, same source, same prompt, same seed, but `--image-strength 0.35`
  instead of `--mask-path`

Published artifacts:

- [full contact sheet](zimage_inpaint_contact_sheet.png)
- [engine crop contact sheet](zimage_inpaint_engine_crop_contact_sheet.png)
- [latent baseline output](zimage_latent_engine_q8.png)
- [native inpaint output](zimage_inpaint_engine_q8.png)
- [stats](zimage_inpaint_stats_m5max.json)
- [commands](zimage_inpaint_command_log.md)

What the sheet proves:

- the new public route is not a silent latent fallback;
- the same prompt and seed can now be driven by a mask-aware native inpaint path;
- the accepted engine case shows a visibly stronger edit concentrated in the masked thruster region.

Measured on an M5 Max:

- latent baseline: `2.78s` generation, `5.99s` wall, `11.49 GB` max RSS
- native inpaint: `21.00s` generation, `26.86s` wall, `18.11 GB` max RSS

Direct review outcome:

- accepted for the narrow engine-thruster proof surface
- public claim stays narrow: exact `AbstractFramework/z-image-turbo-8bit` row and the documented
  same-prompt same-seed engine case
