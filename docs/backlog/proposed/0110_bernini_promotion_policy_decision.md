# Proposed: Bernini-R 1.3B promotion policy decision

## Metadata

- Created: 2026-08-13
- Status: Proposed
- Completed: N/A

## ADR status

- Governing ADRs: [ADR 0001](../../adr/0001_runtime_smoke_validation_for_model_routes.md)
- ADR impact: the decision itself is durable policy and should land as a new ADR (promotion
  criteria for rows whose official recipes are model-limited but whose documented tuned recipes
  succeed), with this item recording the execution.

## Context

Bernini stays experimental and fails closed in release validation. The evidence state as of
2026-08-13: eight of ten upstream public example rows accepted (one at mid profile), and the two
remaining rows (`v2v_case3`, `r2v_case2`) oracle-proven to fail at their official recipes in
both implementations while documented tuned recipes recover them substantially — `v2v_case3`
fully (recipe-parity-proven quadruped) and `r2v_case2` partially (cat-ear elements, chest
branding, drinking motion; the fabric tee never binds at 1.3B). Full-profile `ads2v`
(1280x672/121f) and full-profile oracle runs are recorded as unverifiable scope on the
validation host. See the
[parity matrix](../../assets/validation/bernini-r-1.3b-2026-08-04/official_example_parity_matrix.md).

## Proposed work

1. Decide and record as an ADR: does "matches or exceeds official-1.3B at matched settings, with
   tuned recipes documented" qualify a row for promotion, or do tuned-recipe rows stay
   documented limits?
2. Apply the decision to the release-validation gate (promote Bernini or keep fail-closed with
   the recorded rationale) and update `docs/bernini.md`'s experimental banner accordingly.
3. Close or re-scope backlog 0106 against the current evidence in the same pass.

## Validation expectations

- ADR merged; release-validation behavior matches the decision; docs banner consistent.
- Backlog 0106 state reconciled with a completion or supersession report.

## References

- Parity matrix and bundle (docs/assets/validation/bernini-r-1.3b-2026-08-*)
- `docs/bernini.md`
- Backlog 0106, 0107
