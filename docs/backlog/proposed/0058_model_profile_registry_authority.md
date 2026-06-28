# Proposed: Model profile registry authority

## Metadata
- Created: 2026-06-27
- Status: Proposed
- Completed: N/A

## ADR status
- Governing ADRs: ADR 0002
- ADR impact: Needs an ADR or ADR revision before implementation, because this would establish a
  durable cross-family source of truth.

## Context
The 0051-0057 audit work fixed confirmed local drift in prepare routing and inference-step
defaults, but the same information is still distributed across `ModelConfig`, CLI defaults,
prepare backend selection, task inference, validation metadata, and weight loading policy.

## Problem
Every new model family can reintroduce drift unless aliases, canonical family identity, default
inference parameters, task capabilities, prepare backend identity, weight definition, and loader
trust/precision policy are owned by one enforceable model profile authority.

## What we want to do
Design a staged model profile registry that can be consumed by CLI parsing, Python API defaults,
`mlxgen prepare`, task inference, metadata, and validation tooling without importing heavyweight
model modules at startup.

## Why
The confirmed 0053 and 0054 bugs were symptoms of fragmented model identity/default ownership.
Local fixes are enough for the current release hardening, but continued model growth needs one
auditable contract.

## Requirements
- Write or revise an ADR that defines the registry ownership boundary.
- Keep startup imports lightweight and avoid importing model weights or heavy runtime modules.
- Make the registry authoritative for aliases, canonical family, default steps, capabilities,
  prepare backend id, and loader trust/precision policy.
- Migrate one or two low-risk consumers first before broad rewrites.

## Promotion criteria
Promote after the next model-family addition or default/routing bug shows the current staged
resolver is no longer sufficient, or when an ADR spike proves the registry can avoid import cycles.

## Non-goals
- Do not force a broad registry migration into a release-hotfix branch.
- Do not remove `ModelConfig` unless the ADR proves a cleaner migration path.
