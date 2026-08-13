# Proposed: Bernini-R 1.3B selective 8-bit package

## Metadata

- Created: 2026-08-13
- Status: Proposed
- Completed: N/A

## ADR status

- Governing ADRs: [ADR 0001](../../adr/0001_runtime_smoke_validation_for_model_routes.md),
  [ADR 0002](../../adr/0002_no_silent_automatic_fallbacks.md)
- ADR impact: none expected; the existing fail-closed quantization stance stays until the gates
  below pass.

## Context

Bernini-R 1.3B runs BF16-only. Earlier validation rejected blanket 4-bit (packed-transformer
divergence, overexposed output) and exposed the conservative Wan 8-bit predicate as a no-op for
this transformer: `_is_q8_sensitive_transformer_path` excludes every attention and FFN linear, so
nominal q8 quantized zero Bernini layers while reporting q8. `--quantize` therefore fails closed
for every value (`src/mflux/models/wan/variants/wan_bernini.py`).

The 2026-08-12 parity work built the tooling a real selective policy needs: a CPU-float32
official-code oracle (`untracked/oracle/oracle_run.py`), a step-0 tensor probe
(`untracked/oracle/step0_probe_official.py` / `step0_probe_mlx.py` / `step0_compare.py`), and a
two-package same-seed output comparison (`untracked/oracle/verify_bf16_package.py`) that already
gates the published BF16 package (`AbstractFramework/bernini-r-1.3B-diffusers-bf16`,
bit-identical proof). The team publishes 8-bit packages for other models under the same
namespace, so an 8-bit Bernini package is the natural next tier.

## Proposed work

1. Design a selective q8 policy for the Bernini renderer transformer: start from quantizing FFN
   linears while keeping attention projections, condition embedder, `proj_out`, and the FP32
   keep-set at their current precision; iterate per evidence.
2. Gate every candidate through, in order: step-0 tensor probe deltas vs BF16, reduced-profile
   same-seed output comparison vs BF16, and full-profile contact-sheet review on at least the
   accepted `mv2v` and `ads2v` rows.
3. If a candidate passes, package it (diffusers layout, quantization documented per tensor
   group), publish under `AbstractFramework/`, add a revision-pinned catalog entry mirroring
   `bernini-r-1.3b-bf16`, and lift the `--quantize` rejection only for the published package
   route.
4. If no candidate passes, record the attempted policies and measured deltas here and keep the
   fail-closed stance.

## Validation expectations

- Step-0 probe cosine and relative-L2 per guidance branch recorded for the shipped policy.
- Reduced-profile same-seed comparison and full-profile contact sheets committed under
  `validation_outputs/` with an acceptance note.
- Catalog tests cover the new entry; `--quantize` behavior remains fail-closed for every
  non-published configuration.

## References

- `docs/bernini.md` (Precision And Quantization)
- `src/mflux/models/wan/weights/wan_weight_definition.py` (`quantization_predicate`,
  `BERNINI_TRANSFORMER_PRECISION_POLICY_ID`)
- Backlog 0106 (trajectory parity), 0107 (larger-model scope gate)
