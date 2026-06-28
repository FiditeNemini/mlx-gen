# Completed: Qwen edit parity expansion

## Metadata

- Created: 2026-05-28
- Status: Completed
- Completed: 2026-06-22

## ADR status

- Governing ADRs: [ADR 0001](../../adr/0001_runtime_smoke_validation_for_model_routes.md),
  [ADR 0002](../../adr/0002_no_silent_automatic_fallbacks.md),
  [ADR 0003](../../adr/0003_runtime_truth_vs_consumer_convenience.md)
- ADR impact: No new ADR was required. The shipped Qwen surface stays explicit, route-specific,
  and fail-closed under the existing generation contract.

## Outcome

Completed as the current production-grade Qwen route expansion.

MLX-Gen now ships:

- the first-class [Qwen route matrix](../../qwen-route-matrix.md);
- exact masked edit on `qwen.inpaint`;
- exact structured control on `qwen.control`;
- exact base-Qwen control-inpaint on `qwen.control-inpaint`;
- exact public route proofs for Qwen 2511 multi-reference, reframe, and outpaint under the current
  LoRA contract.

The important design choice was not to invent another public `qwen.edit-plus` route id. MLX-Gen
already exposes the user-visible multi-image behavior through `qwen.multi-reference`, so the
route-matrix and capability contract document that mapping directly.

## Closing code reality

- `mlxgen generate` now exposes the Qwen route family through explicit capability rows:
  `qwen.text`, `qwen.latent`, `qwen.edit`, `qwen.inpaint`, `qwen.multi-reference`,
  `qwen.reframe`, `qwen.outpaint`, `qwen.control`, and `qwen.control-inpaint`.
- The unified router injects the exact validated Qwen ControlNet sidecars and rejects conflicting
  `--controlnet-model` values.
- The shipped `guidance=1` Lightning proof path now skips inactive negative-prompt work and records
  only the effective negative prompt in metadata.
- File-backed source/mask condition caches now invalidate when the input files change in place.
- Qwen edit-family LoRA proof now covers the exact public routes users actually reach through
  `mlxgen generate`, not only the original single-image edit row.

## Validation

Accepted public proof assets:

- [Qwen masked edit report](../../assets/validation/qwen-inpaint-2026-06-15/qwen2511_q8_inpaint_lightning_report.md)
- [Qwen structured-control command log](../../assets/validation/qwen-control-2026-06-15/qwen_q8_control_lightning_command_log.md)
- [Qwen base control-inpaint report](../../assets/validation/qwen-control-inpaint-2026-06-21/qwen_control_inpaint_report.md)
- [Qwen route-completion report](../../assets/validation/lora-route-expansion-2026-06-22/lora_route_expansion_report.md)

Representative proof sheets:

- [Qwen 2511 masked edit Lightning sheet](../../assets/validation/qwen-inpaint-2026-06-15/qwen2511_q8_inpaint_lightning_contact_sheet.png)
- [Qwen structured-control sheet](../../assets/validation/qwen-control-2026-06-15/qwen_q8_control_lightning_contact_sheet.png)
- [Qwen base control-inpaint sheet](../../assets/validation/qwen-control-inpaint-2026-06-21/qwen_control_inpaint_contact_sheet.png)
- [Qwen 2511 multi-reference LoRA sheet](../../assets/validation/lora-route-expansion-2026-06-22/qwen2511_q8_multi_reference_multiangle_exact_contact_sheet.png)
- [Qwen 2511 reframe LoRA sheet](../../assets/validation/lora-route-expansion-2026-06-22/qwen2511_q8_reframe_multi_angle_exact_contact_sheet.png)
- [Qwen 2511 outpaint LoRA sheet](../../assets/validation/lora-route-expansion-2026-06-22/qwen2511_q8_outpaint_multiangle_exact_contact_sheet.png)

Focused tests now cover:

- Qwen route selection and fail-closed option handling;
- mask/control route separation;
- exact sidecar enforcement;
- Qwen prompt-path behavior at `guidance=1`;
- current public validation-profile resolution.

## Remaining boundary

- Broader Qwen control families, layered composition, or new edit pipelines should land as new
  bounded items if they become important.
- The current public Qwen claim stays exact-row oriented. Future expansion should only promote
  another Qwen route after it has its own accepted proof bundle.

## Related backlog items

- [0007 LoRA capability matrix and strict application](0007_lora_capability_matrix_and_strict_application.md)
- [0019 First-class I2I modes and outpaint/reframe UX](../planned/0019_first_class_i2i_modes_and_outpaint_reframe.md)
- [0045 Z-Image ControlNet follow-up](../proposed/0045_zimage_controlnet_followup.md)
