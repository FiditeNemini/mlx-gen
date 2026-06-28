# Completed: LoRA capability matrix and strict application

## Metadata

- Created: 2026-05-28
- Status: Completed
- Completed: 2026-06-22

## ADR status

- Governing ADRs: [ADR 0002](../../adr/0002_no_silent_automatic_fallbacks.md),
  [ADR 0003](../../adr/0003_runtime_truth_vs_consumer_convenience.md)
- ADR impact: No new ADR was required. The shipped design stays inside the existing capability,
  validation, and fail-closed routing contract.

## Outcome

Completed as the production-support boundary for the current MLX-Gen LoRA surface.

MLX-Gen now:

- surfaces route-level LoRA truth through `supports_lora`, `lora_status`,
  `lora_target_roles`, and `lora_validation_profile`;
- rejects unsupported LoRA requests before model dispatch instead of letting constructors silently
  ignore them;
- records structured LoRA application reports in generated metadata;
- keeps `mlxgen prepare --lora-paths/--lora-scales` fail-closed until bake/export equivalence is
  deliberately proven for a family;
- promotes exact validated rows only when the accepted route has a published model-backed A/B proof.

## Closing code reality

- `src/mflux/task_inference.py` now owns the public LoRA capability contract.
- `src/mflux/lora_validation_registry.py` now records exact public validation rows by
  model/package and route.
- `src/mflux/models/common/lora/mapping/lora_loader.py` now fails on the real error cases that
  matter to users: unreadable files, corrupt files, zero matched keys, zero applied targets,
  missing A/B matrices, target-path misses, and matrix-shape mismatches.
- `src/mflux/models/common/lora/lora_compatibility.py` now rejects known cached model-card
  mismatches such as FLUX.2-dev adapters on FLUX.2 Klein.
- Generated metadata now records `lora_application_reports`, `lora_applied_file_count`, and
  `lora_applied_target_count`.

Current exact public validated rows include:

- `AbstractFramework/qwen-image-edit-8bit` on `qwen.edit`
- `AbstractFramework/qwen-image-edit-2509-8bit` on `qwen.edit`
- `AbstractFramework/qwen-image-edit-2511-8bit` on `qwen.edit`, `qwen.inpaint`,
  `qwen.multi-reference`, `qwen.reframe`, and `qwen.outpaint`
- `AbstractFramework/qwen-image-8bit` on `qwen.text`, `qwen.latent`, `qwen.control`, and
  `qwen.control-inpaint`
- `AbstractFramework/qwen-image-2512-8bit` on `qwen.text`
- `AbstractFramework/z-image-turbo-8bit` on `z-image.text`
- `AbstractFramework/ernie-image-turbo-8bit` on `ernie-image.text` and `ernie-image.latent`
- `AbstractFramework/flux.2-klein-9b-8bit` on `flux2.edit` and `flux2.multi-reference`
- `AbstractFramework/flux.2-klein-base-4b-8bit` on `flux2.outpaint`
- all current Wan q8 public video rows

That is the public support claim. Other rows may still surface as `mapped-unvalidated` or
`unsupported`, but they are not part of the production contract until they have their own accepted
proof row.

## Validation

Focused automated coverage now locks:

- capability surfacing;
- unsupported-route rejection;
- strict scale-count handling;
- exact validation-profile resolution;
- FLUX.2 runtime guidance guards on direct model use.

Accepted published proof bundles:

- [LoRA route expansion report](../../assets/validation/lora-route-expansion-2026-06-22/lora_route_expansion_report.md)
- [LoRA route expansion command log](../../assets/validation/lora-route-expansion-2026-06-22/lora_route_expansion_command_log.md)
- [LoRA route expansion stats](../../assets/validation/lora-route-expansion-2026-06-22/lora_route_expansion_stats_m5max.json)
- [Wan video LoRA route matrix](../../assets/validation/wan-lora-2026-06-11/wan_video_lora_route_matrix.jpg)

Representative exact-row artifacts:

- [base Qwen latent realism A/B](../../assets/validation/lora-route-expansion-2026-06-22/qwen_q8_latent_studio_cfg_auto_contact_sheet.png)
- [Qwen 2511 multi-reference A/B](../../assets/validation/lora-route-expansion-2026-06-22/qwen2511_q8_multi_reference_multiangle_exact_contact_sheet.png)
- [Qwen 2511 reframe A/B](../../assets/validation/lora-route-expansion-2026-06-22/qwen2511_q8_reframe_multi_angle_exact_contact_sheet.png)
- [Qwen 2511 outpaint A/B](../../assets/validation/lora-route-expansion-2026-06-22/qwen2511_q8_outpaint_multiangle_exact_contact_sheet.png)
- [FLUX.2 Klein 9B multi-reference A/B](../../assets/validation/lora-route-expansion-2026-06-22/flux2_klein9b_q8_multiref_exact_contact_sheet.png)
- [FLUX.2 Klein base 4B outpaint A/B](../../assets/validation/lora-route-expansion-2026-06-22/flux2_klein_base4b_q8_outpaint_route_exact_contact_sheet.png)

## Remaining boundary

- FLUX.2-dev adapter support remains a separate item because Klein proof does not justify broad
  FLUX.2-dev claims.
- Bonsai packed-runtime LoRA remains a separate architectural problem.
- Future adapter-family or model-family expansion should land as new bounded items instead of
  reopening this contract item.

## Related backlog items

- [0008 Qwen edit parity expansion](0008_qwen_edit_parity_expansion.md)
- [0033 Video LoRA support for T2V and I2V](0033_video_lora_for_t2v_i2v.md)
- [0034 FLUX.2-dev multi-angle LoRA support](../planned/0034_flux2_dev_multi_angle_lora_support.md)
- [0038 Bonsai packed-runtime LoRA support](../proposed/0038_bonsai_packed_lora_runtime_support.md)
