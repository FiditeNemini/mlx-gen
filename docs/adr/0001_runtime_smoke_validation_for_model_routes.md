# ADR 0001: Runtime Smoke Validation For Model Routes

Status: Accepted.

## Context

MLX-Gen model ports often require several layers to agree: model-family detection, config
resolution, weight definitions, transformer dimensions, VAE latent shape, scheduler policy, CLI
defaults, and output writing. Unit tests that check routing or tensor helper shapes can pass while
a real source checkpoint still fails during generation.

A Wan2.2 A14B text-to-video run exposed this failure mode: an unvalidated route could load A14B
weights through a TI2V-5B-shaped runtime and fail at the first transformer convolution because the
latents had 48 channels while the loaded A14B patch embedding expected 16 channels.

## Decision

Any new model route, new model-family alias, or materially different architecture variant must have
at least one model-backed runtime smoke proof before MLX-Gen describes it as working or release
ready.

The smoke proof must exercise the actual selected checkpoint, not only mocks or shape-only helper
tests. For generation backends, it must produce a real output artifact or fail with an intentional
and documented unsupported-state error. If a full-quality run is too expensive, a tiny smoke run is
acceptable, but the backlog and docs must clearly label it as wiring validation rather than quality
validation.

## Consequences

### Positive

- Model-family misrouting, wrong latent channel counts, and wrong VAE/transformer pairing are
  caught before a release claim.
- Backlog completion reports become auditable because they include a command, output path, and
  validation result.
- Large or gated models can still progress incrementally as long as unvalidated surfaces are named
  honestly.

### Negative

- New model ports take longer because at least one real source-checkpoint run is required.
- Very large video models may require a small smoke run and a separate quality-validation backlog
  item rather than a single completion claim.

### Neutral

- Shape tests, config tests, and mocked CLI tests remain necessary, but they are not sufficient
  proof of model support.

## Enforcement

- Backlog items for new model routes must cite this ADR in their ADR status.
- Completion or release notes for a new model route must include the model-backed smoke command,
  output artifact path, and whether the proof is wiring-only or quality-grade.
- Public docs must not say a route is fully supported when the only evidence is a mock, helper
  shape test, or successful weight-load test.
- Correctness-critical model resolution must fail closed instead of silently loading a remote
  checkpoint through a default config when the architecture cannot be inferred.

## Validation

Compliance is validated by:

- focused unit tests for model-family detection and config selection;
- a model-backed smoke run using the real selected checkpoint;
- preserved output artifacts under `validation_outputs/` or another named location;
- backlog `Recent validation` entries that record the exact command and observed result.

## Backlog links

- Originating work: [0012 Wan2.2 A14B T2V/I2V support](../backlog/completed/0012_wan_a14b_t2v_i2v_support.md)

## Related

- `src/mflux/models/wan/cli/wan_generate.py`
- `src/mflux/models/common/config/model_config.py`
- `tests/cli/test_mlx_gen_router.py`
- `tests/wan/test_wan_a14b_config.py`
