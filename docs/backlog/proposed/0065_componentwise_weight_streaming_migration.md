# Proposed: Component-wise weight streaming migration

## Metadata
- Created: 2026-06-27
- Status: Proposed
- Completed: N/A

## ADR status
- Governing ADRs: None
- ADR impact: May need an ADR if the migration establishes a cross-family loader contract or
  rollback policy.

## Context
Planned item [0063](../planned/memory/0063_componentwise_model_loading_memory_policy.md) moved the
safe runtime memory policy earlier by applying MLX cache limits before model construction across
CLIs, but it is not complete until quantitative startup-memory evidence exists. It did not migrate
generic image and restoration initializers to Wan-style component-wise load/apply/clear.

Wan already streams component loading and clears each component after application. Most other
families still load all components into one `LoadedWeights` object before model construction,
weight application, quantization, and LoRA handling.

## Problem
Startup peaks can still include raw loaded component weights, initialized modules, quantization
transients, and adapter work at the same time. This is likely the next meaningful startup-memory
reduction after the pre-construction cache policy fix.

## What we want to do
Design and prove an incremental component-wise migration path for non-Wan families without
changing quantization semantics, validation checks, prepared package layout, or generated quality.

## Why
The memory audit found this as a real issue, but the adversarial pass also found that a naive broad
helper is risky: Qwen, Flux, FIBO, ERNIE, Z-Image, and SeedVR2 each have family-specific
construction, validation, and quantization assumptions.

## Requirements
- Preserve existing `WeightLoader.load()` and `WeightApplier.apply_and_quantize()` paths until a
  family migration is proven.
- Reuse Wan's load/apply/delete/cache-clear pattern where it fits.
- Keep family initializers responsible for component order, validation, model factories, and
  quantization predicates.
- Add rollback or feature-gate guidance if the first migration touches release-critical routes.
- Measure process memory, not only MLX allocator memory, before making public claims.

## Promotion criteria
Promote when profiling shows startup peak remains a practical blocker after the remaining planned
memory items 0060-0064, or when a specific family such as SeedVR2 prepared packages needs the
memory reduction for a supported profile.

Do not promote this item solely from the SeedVR2 1280px image measurements in
`validation_outputs/memory/real_generation_20260627_seedvr2_image_1280_tiling_r2/`. Those runs
show VAE spatial encode tiling reduces MLX peak, while cache-control and startup policy do not.
This migration should remain tied to measured startup or first-step weight-loading overlap.

## Suggested first target
SeedVR2 prepared safetensors are the likely first candidate because video restore is memory
sensitive and the family already has explicit safety budgets. Official `.pth` checkpoints may still
require whole-checkpoint loading and should be treated separately.

## Non-goals
- Do not rewrite every initializer in one pass.
- Do not change prepared checkpoint layout or model-card semantics.
- Do not hide family-specific validation behind a generic abstraction.
- Do not accept output-quality drift to reduce startup memory.

## Validation ideas
- Fake-component tests proving load/apply/delete/cache-clear order.
- Family-specific initializer tests for the first migrated backend.
- Prepared-package q8/q4 smoke for the migrated family.
- Physical-process memory comparison for model construction through first denoise/upscale step.
