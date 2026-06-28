# ADR 0003: Runtime Truth Versus Consumer Convenience

Status: Accepted.

## Context

MLX-Gen now exposes richer route-level capability surfaces: task and mode resolution, route
capabilities, exact validation rows, LoRA support state, mask support, structured-control support,
and fail-closed compatibility checks. Those signals are consumed by higher-level applications such
as AbstractVision.

Without a clear boundary, the runtime and the consumer can drift into overlapping responsibilities.
That creates two risks:

- the runtime becomes a recommendation and curation layer instead of staying the authoritative
  execution boundary;
- the consumer guesses about adapter/model compatibility or route legality instead of reading the
  runtime contract.

This matters now because MLX-Gen already has route-specific Qwen mask/control behavior, exact Wan
video LoRA rows, and model-family-specific compatibility checks. Those should not be re-expressed
as convenience heuristics inside the runtime itself.

## Decision

MLX-Gen is the authoritative producer of runtime truth. Higher-level applications are the owners of
convenience and curation.

In MLX-Gen, the authoritative runtime contract includes:

- exact route identity and task/mode legality;
- capability reporting;
- fail-closed model and adapter compatibility checks when evidence exists;
- validation status and exact validation-profile ids;
- runtime-required parameters and route-specific constraints.

Higher-level consumer layers such as AbstractVision should own:

- curated model-to-adapter pairing;
- recommended presets and user-facing defaults;
- ranking, filtering, and presentation of available adapters;
- convenience abstractions that span several backends or providers.

MLX-Gen may document recommended exact routes and validated examples, but it must not become a
general adapter recommendation engine or marketplace abstraction.

## Consequences

### Positive

- One layer remains the source of truth for whether a request is legal and proven.
- Higher-level products can build UI and workflow convenience without weakening runtime contracts.
- Capability and validation metadata become reusable instead of being duplicated as hardcoded
  product logic.

### Negative

- Some user-facing convenience stays out of the runtime even when it would be easy to add locally.
- Higher-level integrations must actively consume MLX-Gen capability and validation fields instead
  of assuming defaults.

### Neutral

- MLX-Gen can still ship exact worked examples and validated adapter commands as documentation.
- Cross-repo integration lag is possible; that lag should be handled as consumer adoption work, not
  by silently broadening MLX-Gen responsibility.

## Enforcement

- New runtime features in MLX-Gen must expose machine-readable route truth through capabilities,
  validation metadata, explicit errors, or generated metadata.
- Code review should reject MLX-Gen changes that try to add broad heuristic model-to-adapter
  recommendation logic when the same outcome belongs in a higher-level integration layer.
- Backlog items that add route capability, route validation, adapter checks, or consumer-boundary
  docs should cite this ADR when the ownership split matters.
- Public docs may recommend exact validated routes, but they must not describe MLX-Gen as the owner
  of cross-provider convenience or curation.

## Validation

Compliance is validated by:

- `mlxgen capabilities` exposing route truth such as `supports_lora`, `supports_mask`,
  `supports_control_image`, `lora_status`, and validation-profile ids;
- focused tests that unsupported route/option combinations fail closed instead of degrading
  silently;
- Python integration and FAQ docs describing the runtime-versus-consumer ownership split
  consistently;
- backlog review confirming that cross-repo consumer adoption work is not mislabeled as core
  MLX-Gen runtime work.

## Backlog links

- Related runtime expansion: [0008 Qwen edit parity expansion](../backlog/completed/0008_qwen_edit_parity_expansion.md)
- Related runtime validation: [0007 LoRA capability matrix and strict application](../backlog/completed/0007_lora_capability_matrix_and_strict_application.md)
- Recurrent hygiene: [0017 Backlog release-state hygiene](../backlog/recurrent/0017_backlog_release_hygiene.md)

## Related

- [ADR 0001: Runtime Smoke Validation For Model Routes](0001_runtime_smoke_validation_for_model_routes.md)
- [ADR 0002: No Silent Automatic Fallbacks](0002_no_silent_automatic_fallbacks.md)
- [docs/python-integration.md](../python-integration.md)
- [docs/lora.md](../lora.md)
