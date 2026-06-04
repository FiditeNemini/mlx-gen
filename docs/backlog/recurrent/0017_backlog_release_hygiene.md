# Recurrent: Backlog release-state hygiene

## Metadata

- Created: 2026-06-04
- Status: Recurrent
- Completed: N/A
- Cadence: After each release, after large validation runs, and before any planned/proposed split
  that changes priorities.

## ADR status

- Governing ADRs: None
- ADR impact: None. Escalate to ADR only if this process creates a durable engineering policy
  beyond backlog maintenance.

## Purpose

Keep backlog state aligned with shipped code, public docs, validation evidence, and release claims.
The backlog is useful only when planned work names what remains rather than what already shipped.

## Trigger

- A release is tagged or published.
- A model is uploaded or a model card is materially rewritten.
- A long validation run changes a model-family support claim.
- A planned item is mostly completed but still has residual follow-up work.
- A proposed item becomes a concrete correctness or ADR-alignment issue.

## Checklist

- Recount planned, proposed, completed, deprecated, and recurrent item files.
- Check global four-digit ID uniqueness across all backlog lifecycle folders.
- Move completed work to `completed/` with a completion report instead of leaving shipped work as
  planned.
- Narrow partially completed planned items so their remaining scope is explicit.
- Promote proposed items only when evidence shows urgency, blocking risk, or a clear mandate.
- Fix stale lifecycle links after moves.
- Update `overview.md` counts, ledgers, next recommended work, and planning notes in the same pass.
- Confirm planned work cites relevant ADRs or explicitly records why none applies.

## Latest run

- 2026-06-04: Post-0.18.9 hygiene promoted LoRA strictness to planned work, narrowed Wan A14B
  boundary-memory scope to the remaining full-size retry, updated Wan q8 integrity/performance
  wording for the shipped 0.18.9 guardrail baseline, and normalized completed-item reports.
- 2026-06-04: Pre-0.18.10 hygiene checked the current taskless/capability planner against code and
  docs, added release-readiness evidence to completed item 0020, narrowed planned item 0019 to the
  remaining first-class FLUX.1 Fill outpaint/reframe adapter, and confirmed item 0016 still gates
  full-size Wan A14B q8 readiness claims.
