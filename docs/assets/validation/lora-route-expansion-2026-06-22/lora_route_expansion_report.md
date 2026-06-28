# LoRA Route Expansion Report

Accepted production bundle for the June 22 exact image LoRA rows that remain public and
production-supported after the 2026-06-23 bundle review.

The current Z-Image Turbo q8 latent row is documented in a separate dedicated bundle:
[Z-Image latent LoRA report](../zimage-latent-lora-2026-06-24/zimage_latent_lora_report.md).

## Accepted Rows

| Route | Exact model | Adapter set | Result | Contact sheet |
| --- | --- | --- | --- | --- |
| `qwen.text` | `AbstractFramework/qwen-image-8bit` | `flymy-ai/qwen-image-realism-lora:flymy_realism.safetensors` | `PASS` | [Base Qwen text realism A/B](qwen_q8_text_realism_ab_contact_sheet.png) |
| `qwen.latent` | `AbstractFramework/qwen-image-8bit` | `prithivMLmods/Qwen-Image-Studio-Realism:qwen-studio-realism.safetensors` | `PASS` | [Base Qwen latent realism A/B](qwen_q8_latent_studio_cfg_auto_contact_sheet.png) |
| `qwen.reframe` | `AbstractFramework/qwen-image-edit-2511-8bit` | `lightx2v/Qwen-Image-Edit-2511-Lightning:Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors` + `fal/Qwen-Image-Edit-2511-Multiple-Angles-LoRA:qwen-image-edit-2511-multiple-angles-lora.safetensors` | `PASS` | [Qwen 2511 reframe A/B](qwen2511_q8_reframe_multi_angle_exact_contact_sheet.png) |
| `qwen.outpaint` | `AbstractFramework/qwen-image-edit-2511-8bit` | `lightx2v/Qwen-Image-Edit-2511-Lightning:Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors` + `fal/Qwen-Image-Edit-2511-Multiple-Angles-LoRA:qwen-image-edit-2511-multiple-angles-lora.safetensors` | `PASS` | [Qwen 2511 outpaint A/B](qwen2511_q8_outpaint_multiangle_exact_contact_sheet.png) |
| `qwen.multi-reference` | `AbstractFramework/qwen-image-edit-2511-8bit` | `lightx2v/Qwen-Image-Edit-2511-Lightning:Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors` + `fal/Qwen-Image-Edit-2511-Multiple-Angles-LoRA:qwen-image-edit-2511-multiple-angles-lora.safetensors` | `PASS` | [Qwen 2511 multi-reference A/B](qwen2511_q8_multi_reference_multiangle_exact_contact_sheet.png) |
| `flux2.multi-reference` | `AbstractFramework/flux.2-klein-9b-8bit` | `dx8152/Flux2-Klein-9B-Migration:Klein-Migration.safetensors` | `PASS` | [FLUX.2 Klein 9B multi-reference A/B](flux2_klein9b_q8_multiref_exact_contact_sheet.png) |
| `flux2.outpaint` | `AbstractFramework/flux.2-klein-base-4b-8bit` | `fal/flux-2-klein-4B-outpaint-lora:flux-outpaint-lora.safetensors` | `PASS` | [FLUX.2 Klein base 4B outpaint A/B](flux2_klein_base4b_q8_outpaint_route_exact_contact_sheet.png) |
| `ernie-image.latent` | `AbstractFramework/ernie-image-turbo-8bit` | `reverentelusarca/ernie-image-elusarca-anime-style-lora:ernie-anime-v1.safetensors` | `PASS` | [ERNIE latent anime A/B](ernie_turbo_q8_latent_anime_style_ab_contact_sheet.png) |

## Current Boundary

- The Z-Image Turbo q8 latent row is no longer part of this June 22 bundle because it now has a
  dedicated June 24 same-source A/B bundle with a clearer latent style-transfer proof.
- `AbstractFramework/flux.2-klein-base-4b-8bit` outpaint already works without a LoRA. The
  accepted `flux2.outpaint` row exists to prove the dedicated outpaint adapter on that exact route,
  not to prove that outpaint itself requires a LoRA.

## What Was Proved

- Base Qwen Image q8 now has exact validated public rows on `qwen.text`, `qwen.latent`,
  `qwen.control`, and `qwen.control-inpaint`.
- Z-Image Turbo q8 now has exact public rows on `z-image.text` and `z-image.latent`, with the
  latent row documented in the separate June 24 bundle linked above.
- Qwen Image Edit 2511 q8 now has exact validated public rows on the route family users actually
  reach through `mlxgen generate`: `qwen.edit`, `qwen.inpaint`, `qwen.reframe`, `qwen.outpaint`,
  and `qwen.multi-reference`.
- FLUX.2 Klein now has exact validated public LoRA rows with honest scope:
  `AbstractFramework/flux.2-klein-9b-8bit` on `flux2.edit` and `flux2.multi-reference`, and
  `AbstractFramework/flux.2-klein-base-4b-8bit` on `flux2.outpaint`.
- ERNIE Image Turbo q8 has exact public rows on `ernie-image.text` and `ernie-image.latent`.

That is the production-supported boundary. Other LoRA-capable rows may still surface as
`mapped-unvalidated` or `unsupported` through `mlxgen capabilities`, but they are not part of the
public support claim until they have their own accepted A/B proof.

## Reading The Sheets

- Every sheet now includes the model, route, exact prompt, and key parameters in readable text.
- Every accepted row is a same-seed A/B with the route held fixed and the adapter change made
  explicit in the panel labels.
- The Qwen 2511 reframe/outpaint/multi-reference rows use prompt-matched Lightning-only baselines.
  The promoted column then adds the exact multi-angle adapter on top of the same prompt and seed.
- The FLUX.2 base 4B outpaint sheet uses the generated green outpaint canvas as the first panel so
  the fill area is visible instead of implied.

## Runtime Notes

- The exact commands are in [lora_route_expansion_command_log.md](lora_route_expansion_command_log.md).
- The measured local generation times and loader counts are in
  [lora_route_expansion_stats_m5max.json](lora_route_expansion_stats_m5max.json).
- The late-added Qwen and FLUX rows reuse generated metadata for exact generation times and LoRA
  application counts. Treat those numbers as route-local evidence, not as a cross-model speed
  leaderboard.
