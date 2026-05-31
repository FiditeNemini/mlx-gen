# ADR 0002: No Silent Automatic Fallbacks

Status: Accepted.

## Context

MLX-Gen routes user requests across model families, model variants, local prepared folders, and
large source checkpoints. A silent fallback in model identity or architecture can make a command
appear to run while using the wrong model, wrong config, wrong latent shape, wrong quantization
policy, or wrong task semantics.

Wan2.2 A14B support exposed this risk directly: a request for one Wan variant must never become a
TI2V-5B run because a generic `wan` substring or default config happened to match. The same rule
matters for every backend that chooses an architecture from a user-supplied model name or path.

## Decision

MLX-Gen must fail closed when it cannot resolve the exact requested model identity, backend,
architecture variant, quantization policy, or task contract.

Silent automatic fallbacks are forbidden. Code must not substitute a default model, default
backend, alternate architecture, alternate task mode, or degraded behavior unless the user or
caller explicitly requested that fallback behavior through a named option or API parameter.

Fallback-like resolution is allowed only when it does not change the requested identity or
semantics. For example, deterministic search across known tokenizer subdirectories inside the same
requested checkpoint is layout resolution, not model fallback. Choosing TI2V-5B for an unknown Wan
path, choosing another model family because a string contains a broad keyword, or ignoring an edit
mask by switching to unmasked edit behavior is forbidden.

## Consequences

### Positive

- A successful command means MLX-Gen used the model family and architecture the user asked for.
- Unsupported or ambiguous models fail with actionable errors instead of producing misleading
  outputs.
- Future integrations with AbstractVision can rely on failure as a real signal rather than
  reverse-engineering whether MLX-Gen silently changed behavior.

### Negative

- Some formerly convenient ambiguous local paths must be renamed, given explicit metadata, or
  called with a supported exact alias.
- Experimental ports may fail earlier while model detection and metadata support are completed.

### Neutral

- Explicit user-requested behavior remains valid. For example, a future `--base-model`,
  `--family`, or similarly named option may intentionally tell MLX-Gen how to interpret an
  otherwise ambiguous local folder.
- Same-artifact layout resolution remains valid when it preserves the requested model identity and
  is covered by tests.

## Enforcement

- Model, backend, architecture, task, and quantization resolution code must not return a default
  config after failed inference.
- If a fallback is intentionally added, it must be explicit in the option/API name, documented, and
  covered by tests that prove both the opt-in path and the fail-closed default.
- Error messages for ambiguous model names or paths must state what could not be inferred and how
  to make the request explicit.
- Code review should reject wording or implementation patterns such as "fallback to default model",
  "try base model if unknown", or "silently continue without requested capability" on
  correctness-critical paths.
- Backlog items that add model-family routing or compatibility behavior must cite this ADR.

## Validation

Compliance is validated by:

- focused tests that unknown model names and ambiguous local paths fail instead of selecting a
  default model;
- tests that exact aliases and explicit model metadata still resolve correctly;
- search audits for fallback/default-return code in model routing, task routing, and quantization
  policy code;
- release notes or completion reports that state whether any fallback-like behavior is explicit
  opt-in and how it is tested.

## Backlog links

- Originating work: [0012 Wan2.2 A14B T2V/I2V support](../backlog/completed/0012_wan_a14b_t2v_i2v_support.md)

## Related

- [ADR 0001: Runtime Smoke Validation For Model Routes](0001_runtime_smoke_validation_for_model_routes.md)
- `src/mflux/models/wan/cli/wan_generate.py`
- `src/mflux/models/common/config/model_config.py`
- `tests/cli/test_mlx_gen_router.py`
