# Qwen base control-inpaint proof

Accepted proof surface:

- route: `mlxgen generate --model AbstractFramework/qwen-image-8bit --image ... --mask-path ...`
- resolved capability: `qwen.control-inpaint`
- exact sidecar: `InstantX/Qwen-Image-ControlNet-Inpainting:diffusion_pytorch_model.safetensors`
- fast adapter: `lightx2v/Qwen-Image-Lightning`

Published artifacts:

- [contact sheet](qwen_control_inpaint_contact_sheet.png)
- [engine output](qwen_control_inpaint_engine_lightning_q8.png)
- [repair output](qwen_control_inpaint_repair_lightning_q8.png)
- [stats](qwen_control_inpaint_stats_m5max.json)
- [commands](qwen_control_inpaint_command_log.md)

What the sheet proves:

- same source, mask, prompt, seed, and Lightning adapter across the comparison row;
- the new base-Qwen route stays localized and produces acceptable engine and repair results;
- the route is distinct from `qwen.inpaint` on the edit checkpoint even though the user request
  shape is still `image + mask + prompt`.

Measured on an M5 Max:

- engine row: `17.29s` generation, `24.65s` wall, `34.94 GB` max RSS
- repair row: `17.74s` generation, `23.86s` wall, `34.94 GB` max RSS

Direct review outcome:

- accepted
- public claim is narrow: exact q8 base-Qwen row, exact InstantX inpainting sidecar, and the
  documented Lightning fast path
