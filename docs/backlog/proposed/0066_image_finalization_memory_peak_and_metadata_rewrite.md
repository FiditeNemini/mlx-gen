# Proposed: Image finalization memory peak and metadata rewrite

## Metadata
- Created: 2026-06-29
- Status: Proposed
- Completed: N/A

## ADR status
- Governing ADRs: None
- ADR impact: None unless the fix establishes a new global image-metadata policy or removes
  runtime-memory metadata from the default generated-image contract.

## Context
The current memory track already covers startup, prompt materialization, and retention behavior in
items [0060](../completed/0060_runtime_memory_telemetry_and_manifests.md),
[0061](../completed/0061_prompt_materialization_for_low_ram_release.md), and
[0064](../planned/memory/0064_generation_retention_cleanup.md). A separate user-observed issue
remains at the very end of image generation: process memory spikes while the generated image is
being finalized and written to disk.

This is not the denoising loop itself. The spike appears after generation has logically finished
and before control returns to the caller.

## Current code reality
- `src/mflux/utils/generated_image.py` calls `ImageUtil.save_image(...)` from
  `GeneratedImage.save()`.
- `src/mflux/utils/image_util.py` currently does three full-image save passes for ordinary PNG
  output:
  1. primary `image.save(file_path, format=image_format)`
  2. `_embed_metadata(...)`, which reopens the just-written image and saves it again with EXIF
  3. `MetadataBuilder.embed_metadata(...)`, which reopens the image again and saves it again with
     PNG XMP/IPTC metadata
- `src/mflux/utils/generated_image.py` adds
  `RuntimeMemory.snapshot("image-metadata").to_metadata()` inside `_get_metadata()`.
- `src/mflux/utils/runtime_memory.py` can perform extra end-of-run work during that snapshot,
  including a Darwin helper subprocess for physical-footprint sampling.

## Problem or opportunity
Image finalization currently combines two independent memory-pressure sources:

1. repeated reopen-and-resave behavior for metadata embedding; and
2. runtime-memory snapshot collection during metadata construction.

That creates a plausible end-of-run RSS / physical-footprint spike even when the denoising path is
already done.

## Proposed direction
Treat final image save as a memory-focused follow-up item rather than assuming the current
post-processing cost is acceptable.

The likely fix direction is:
- collapse metadata embedding into one final write pass instead of three image writes;
- avoid reopening the just-saved PNG multiple times;
- decide whether runtime-memory metadata belongs in the default generated-image metadata path or
  should be opt-in / sidecar-only / benchmark-only.

## Why it might matter
This is exactly the part of the run users experience as “memory spikes at the end even though the
image is already generated.” That hurts trust, makes memory graphs look worse than the actual
generation loop, and can blur the boundary between model-runtime pressure and save-pipeline
pressure.

## Promotion criteria
Promote when one of the following is true:
- process memory sampling confirms a meaningful end-of-run save/finalization peak on representative
  PNG image-generation routes; or
- a safe one-pass metadata embedding design is ready and clearly lower-risk than keeping the
  current three-pass save pipeline.

## Validation ideas
- Measure RSS / physical footprint before primary `image.save`, after primary save, after EXIF
  embedding, and after PNG XMP/IPTC embedding.
- Compare the current pipeline against a single-pass metadata-embedding prototype on large PNG
  outputs.
- Re-measure with runtime-memory metadata enabled versus disabled.
- Confirm that metadata readers still recover EXIF, XMP, and IPTC fields after any rewrite.

## Non-goals
- Do not broaden this into generic model-runtime memory tuning.
- Do not remove useful metadata silently without deciding the public metadata contract.
- Do not conflate this save-path issue with prompt-materialization or hidden-state-retention work
  unless measurements prove they overlap materially.

## Guidance for future agents
Start from the save pipeline, not from denoising internals. The highest-signal question is whether
the final PNG write path is doing multiple full-image in-memory rewrites and whether runtime-memory
metadata is worth its cost in the default path.
